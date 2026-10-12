"""Local Streamlit dashboard: review decisions, override them, and approve moves (Phase 6).

Start with ``downloads-organizer dashboard``. It listens on localhost only. Files move
only when you press a Move button and tick the confirmation; nothing is ever deleted.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloads_organizer import database, decision_engine, organizer  # noqa: E402
from downloads_organizer.locking import AlreadyRunning, run_lock  # noqa: E402
from downloads_organizer.config import load_config  # noqa: E402
from downloads_organizer.decision_engine import TYPE_FOLDERS  # noqa: E402
from downloads_organizer.reporter import human_size  # noqa: E402

ACTIONS = ["KEEP", "ARCHIVE", "REVIEW", "CLEANUP_CANDIDATE", "LEAVE"]
ACTION_HELP = {
    "KEEP": "File into Organized/<type folder>",
    "ARCHIVE": "File into Organized/Archive/<type folder>",
    "REVIEW": "Needs your attention",
    "CLEANUP_CANDIDATE": "Probably not needed; you decide whether to delete it yourself",
    "LEAVE": "Leave where it is",
}
FOLDERS = sorted(
    set(TYPE_FOLDERS.values())
    | {f"{decision_engine.ARCHIVE_DIR}/{f}" for f in TYPE_FOLDERS.values()}
    | {decision_engine.REVIEW_DIR, decision_engine.CLEANUP_DIR, decision_engine.DUPLICATES_DIR}
)


def _config_path() -> str | None:
    if "--config" in sys.argv:
        return sys.argv[sys.argv.index("--config") + 1]
    return os.environ.get("SDO_CONFIG")


config = load_config(_config_path())
st.set_page_config(page_title="Downloads Organizer", layout="wide")

if not config.database_path.exists():
    st.title("Smart Downloads Organizer")
    st.info("No data yet. Run `downloads-organizer run` first.")
    st.stop()

conn = database.connect(config.database_path)


def load_decisions() -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT d.id, d.filename, d.original_path, d.size_bytes, d.modified_at, d.extension,
               d.classification_json, x.action, x.destination, x.ai_action, x.rules, x.reasons,
               o.action AS your_action, o.destination AS your_destination,
               p.original_path AS kept_copy
        FROM decisions x
        JOIN documents d ON d.id = x.document_id
        LEFT JOIN user_overrides o ON o.document_id = d.id
        LEFT JOIN documents p ON p.id = x.duplicate_of
        WHERE d.present = 1
        """
    ).fetchall()
    records = []
    for r in rows:
        c = (json.loads(r["classification_json"]) or {}).get("classification", {}) if r["classification_json"] else {}
        records.append({
            "id": r["id"],
            "file": r["filename"],
            "decision": r["action"],
            "destination": r["destination"] or "",
            "your decision": r["your_action"] or "",
            "your folder": r["your_destination"] or "",
            "AI suggested": r["ai_action"] or "",
            "type": c.get("document_type", ""),
            "category": c.get("category", ""),
            "confidence": c.get("confidence"),
            "why": " | ".join(json.loads(r["reasons"])),
            "summary": c.get("summary", ""),
            "size": human_size(r["size_bytes"]),
            "modified": (r["modified_at"] or "")[:10],
            "kept copy": r["kept_copy"] or "",
            "path": r["original_path"],
        })
    return pd.DataFrame.from_records(records)


def save_overrides(before: pd.DataFrame, edited: pd.DataFrame) -> int:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    changed = 0
    for (_, old), (_, new) in zip(before.iterrows(), edited.iterrows()):
        if (old["your decision"], old["your folder"]) == (new["your decision"], new["your folder"]):
            continue
        changed += 1
        if not new["your decision"]:
            conn.execute("DELETE FROM user_overrides WHERE document_id = ?", (int(new["id"]),))
            continue
        conn.execute(
            "INSERT INTO user_overrides (document_id, action, destination, updated_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(document_id) DO UPDATE SET action = excluded.action,"
            " destination = excluded.destination, updated_at = excluded.updated_at",
            (int(new["id"]), new["your decision"], new["your folder"] or None, now),
        )
    conn.commit()
    if changed:
        decision_engine.decide_all(conn, config)
    return changed


def editable_table(df: pd.DataFrame, key: str, columns: list[str]) -> None:
    if df.empty:
        st.success("Nothing here.")
        return
    view = df[["id", *columns, "your decision", "your folder"]].reset_index(drop=True)
    edited = st.data_editor(
        view,
        key=key,
        hide_index=True,
        width="stretch",
        disabled=[c for c in view.columns if c not in ("your decision", "your folder")],
        column_config={
            "id": None,
            "your decision": st.column_config.SelectboxColumn(
                options=[""] + ACTIONS, help="Overrides the rules for this file. Leave blank to use the rules."),
            "your folder": st.column_config.SelectboxColumn(
                options=[""] + FOLDERS, help="Optional. Blank = folder chosen from the document type."),
            "confidence": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if st.button("Save my decisions", key=f"{key}-save", type="primary"):
        n = save_overrides(view, edited)
        st.toast(f"Saved {n} change(s).")
        st.rerun()


st.sidebar.title("Downloads Organizer")
page = st.sidebar.radio("Page", ["Overview", "Review queue", "All files", "Duplicates", "Move files", "History"])
st.sidebar.caption(f"Downloads: `{config.downloads_dir}`")
st.sidebar.caption(f"Organized: `{config.organized_dir}`")

if conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0:
    decision_engine.decide_all(conn, config)
df = load_decisions()

if page == "Overview":
    st.title("Overview")
    last = conn.execute("SELECT MAX(finished_at) FROM scan_runs").fetchone()[0]
    classified = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE present = 1 AND classification_status = 'done'").fetchone()[0]
    dup_bytes = conn.execute(
        """SELECT COALESCE(SUM(d.size_bytes), 0) FROM decisions x JOIN documents d ON d.id = x.document_id
           WHERE x.rules LIKE '%DUPLICATE_COPY%' AND d.present = 1""").fetchone()[0]
    a, b, c, e = st.columns(4)
    a.metric("Files tracked", len(df))
    b.metric("Classified by AI", classified)
    c.metric("Waiting for review", int((df["decision"] == "REVIEW").sum()) if not df.empty else 0)
    e.metric("Space in duplicate copies", human_size(dup_bytes))
    st.caption(f"Last scan: {last or 'never'}")

    if not df.empty:
        left, right = st.columns(2)
        left.subheader("Decisions")
        left.bar_chart(df["decision"].value_counts())
        right.subheader("Categories (classified files)")
        cats = df.loc[df["category"] != "", "category"].value_counts()
        right.bar_chart(cats)
        st.subheader("What each decision means")
        st.table(pd.DataFrame({"decision": ACTIONS, "meaning": [ACTION_HELP[a] for a in ACTIONS]}))

    if st.button("Check for new downloads now", help="Scan, read and classify new files. Nothing is moved."):
        from downloads_organizer.cli import run_pipeline

        try:
            with st.spinner("Scanning, reading and classifying new files..."):
                run_pipeline(config)
        except AlreadyRunning as exc:
            st.warning(str(exc))
        else:
            st.rerun()

elif page == "Review queue":
    st.title("Review queue")
    st.write("Files the organizer won't decide on by itself. Pick a decision for any you know, then save.")
    queue = df[df["decision"] == "REVIEW"]
    reasons = Counter(w.split(" | ")[0] for w in queue["why"]) if not queue.empty else Counter()
    if reasons:
        reason = st.selectbox("Show", ["All"] + [f"{r} ({n})" for r, n in reasons.most_common()])
        if reason != "All":
            queue = queue[queue["why"].str.startswith(reason.rsplit(" (", 1)[0])]
    editable_table(queue, "review", ["file", "why", "AI suggested", "type", "confidence", "size", "path"])

elif page == "All files":
    st.title("All files")
    f1, f2, f3 = st.columns(3)
    actions = f1.multiselect("Decision", ACTIONS, default=["KEEP", "ARCHIVE", "REVIEW", "CLEANUP_CANDIDATE"])
    categories = sorted(c for c in df["category"].unique() if c) if not df.empty else []
    chosen = f2.multiselect("Category", categories)
    search = f3.text_input("Search name or path")
    view = df[df["decision"].isin(actions)] if not df.empty else df
    if chosen:
        view = view[view["category"].isin(chosen)]
    if search:
        view = view[view["path"].str.contains(search, case=False, regex=False)]
    st.caption(f"{len(view)} file(s)")
    editable_table(view, "all", ["file", "decision", "destination", "type", "confidence", "why", "path"])

elif page == "Duplicates":
    st.title("Duplicate copies")
    st.write("Byte-identical files. One copy is kept; the others are cleanup candidates. "
             "Nothing is deleted. Moving them puts them in `Cleanup_Candidates/Duplicates` for you to check.")
    dups = df[df["kept copy"] != ""] if not df.empty else df
    if dups.empty:
        st.success("No duplicate copies.")
    else:
        st.dataframe(dups[["file", "kept copy", "size", "path"]], hide_index=True, width="stretch")

    st.subheader("Possible versions of the same document")
    st.caption("Different files whose text is largely the same: drafts, re-exports, tailored resumes. "
               "Nothing is changed automatically; keep the ones you need.")
    similar = conn.execute(
        """SELECT g.group_id, g.similarity_score, d.filename, d.modified_at, d.original_path
           FROM duplicate_groups g JOIN documents d ON d.id = g.document_id
           WHERE g.duplicate_type = 'similar' AND d.present = 1
           ORDER BY g.group_id, d.modified_at DESC"""
    ).fetchall()
    if not similar:
        st.success("No near-duplicates found.")
    else:
        st.dataframe(pd.DataFrame([{
            "group": r["group_id"], "similarity": r["similarity_score"], "file": r["filename"],
            "modified": (r["modified_at"] or "")[:10], "path": r["original_path"],
        } for r in similar]), hide_index=True, width="stretch")

elif page == "Move files":
    st.title("Move files")
    st.write("Choose which decisions to carry out. You can see the full plan before anything moves, "
             "and every move can be undone from **History**.")
    actions = st.multiselect("Include", ["KEEP", "ARCHIVE", "REVIEW", "CLEANUP_CANDIDATE"],
                             default=["KEEP", "ARCHIVE"])
    if actions:
        moves, skipped = organizer.plan_moves(conn, config, tuple(actions))
        plan = pd.DataFrame([{
            "file": m.source.name, "decision": m.action,
            "to": str(m.destination.relative_to(config.organized_dir)), "why": m.reason, "from": str(m.source),
        } for m in moves])
        st.subheader(f"{len(moves)} file(s) will move to `{config.organized_dir}`")
        if not plan.empty:
            st.dataframe(plan, hide_index=True, width="stretch")
        if skipped:
            with st.expander(f"{len(skipped)} skipped"):
                st.dataframe(pd.DataFrame([{"path": s.path, "why": s.reason} for s in skipped]), hide_index=True)
        if moves:
            ok = st.checkbox(f"I've checked this plan. Move {len(moves)} file(s).")
            if st.button("Move files", type="primary", disabled=not ok):
                try:
                    with run_lock(config.data_dir, "dashboard move"), st.spinner("Moving and verifying..."):
                        batch, moved, failures = organizer.apply_moves(conn, moves)
                except AlreadyRunning as exc:
                    st.warning(str(exc))
                else:
                    st.success(f"Moved {moved} file(s) (batch {batch}). You can undo this from History.")
                    for move, error in failures:
                        st.error(f"{move.source}: {error}")

elif page == "History":
    st.title("History")
    batches = organizer.batches(conn)
    if not batches:
        st.info("No files have been moved yet.")
    else:
        st.dataframe(pd.DataFrame([dict(b) for b in batches]), hide_index=True, width="stretch")
        undoable = [b["batch_id"] for b in batches if b["completed"]]
        if undoable:
            batch = st.selectbox("Undo a batch", undoable)
            ops = conn.execute("SELECT source_path, destination_path FROM file_operations"
                               " WHERE batch_id = ? AND status = 'completed'", (batch,)).fetchall()
            st.dataframe(pd.DataFrame([{"back to": o[0], "from": o[1]} for o in ops]), hide_index=True,
                         width="stretch")
            ok = st.checkbox(f"Move these {len(ops)} file(s) back")
            if st.button("Undo batch", disabled=not ok):
                try:
                    with run_lock(config.data_dir, "dashboard undo"):
                        restored, problems = organizer.rollback(conn, config, batch)
                except AlreadyRunning as exc:
                    st.warning(str(exc))
                else:
                    st.success(f"Restored {restored} file(s).")
                    for path, problem in problems:
                        st.warning(f"{path}: {problem}")

conn.close()
