"""Combine AI recommendations with deterministic safety rules (Phase 4).

The AI suggests; these rules decide. Every decision carries the rules that
fired and a human-readable reason. Nothing here touches the filesystem except
read-only checks for project folders.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from downloads_organizer import extractors
from downloads_organizer.config import Config, DecisionSettings

KEEP = "KEEP"
ARCHIVE = "ARCHIVE"
REVIEW = "REVIEW"
CLEANUP = "CLEANUP_CANDIDATE"
LEAVE = "LEAVE"  # no move proposed

# document_type -> folder under Organized/ (matches the plan's folder structure)
TYPE_FOLDERS = {
    "Resume or CV": "Career/Resumes",
    "Job description": "Career/Job_Descriptions",
    "Offer letter or employment document": "Career/Offers",
    "Certificate": "Education/Certificates",
    "Academic notes": "Education/Notes",
    "Assignment or examination material": "Education/Assignments",
    "Project documentation": "Projects",
    "Financial document": "Finance",
    "Invoice or receipt": "Finance/Receipts",
    "Government or legal document": "Legal_and_Official",
    "Reference material": "Reference",
    "Temporary or miscellaneous download": "Miscellaneous",
}
ARCHIVE_DIR = "Archive"
REVIEW_DIR = "Needs_Review"
CLEANUP_DIR = "Cleanup_Candidates"
DUPLICATES_DIR = "Cleanup_Candidates/Duplicates"

COPY_MARKER = re.compile(r"\(\d+\)|\bcopy\b|- copy|_copy", re.IGNORECASE)


@dataclass
class Decision:
    document_id: int
    action: str
    destination: str | None
    reasons: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    ai_action: str | None = None
    duplicate_of: int | None = None


def destination_for(action: str, document_type: str | None) -> str | None:
    folder = TYPE_FOLDERS.get(document_type or "", "Miscellaneous")
    return {
        KEEP: folder,
        ARCHIVE: f"{ARCHIVE_DIR}/{folder}",
        REVIEW: REVIEW_DIR,
        CLEANUP: CLEANUP_DIR,
    }.get(action)


class ProjectDetector:
    """Finds files inside software projects, which must not be split up. Read-only."""

    def __init__(self, root: Path, settings: DecisionSettings):
        self.root = root.resolve()
        self.markers = settings.project_markers
        self.name_patterns = [re.compile(p, re.IGNORECASE) for p in settings.project_name_patterns]
        self._cache: dict[Path, bool] = {}

    def _is_project_dir(self, folder: Path) -> bool:
        if folder not in self._cache:
            self._cache[folder] = any(p.search(folder.name) for p in self.name_patterns) or any(
                (folder / marker).exists() for marker in self.markers
            )
        return self._cache[folder]

    def project_of(self, path: str) -> Path | None:
        """The outermost project folder (below the Downloads root) containing ``path``."""
        try:
            relative = Path(path).resolve().relative_to(self.root)
        except ValueError:
            return None
        folder = self.root
        for part in relative.parts[:-1]:
            folder = folder / part
            if self._is_project_dir(folder):
                return folder
        return None


def _primary_copy(members: list[sqlite3.Row], in_project: dict[int, bool]) -> sqlite3.Row:
    """The copy to keep: inside a project, then without "(1)"/"copy", shallowest, oldest."""
    return min(
        members,
        key=lambda d: (
            not in_project[d["id"]],
            bool(COPY_MARKER.search(Path(d["filename"]).stem)),
            Path(d["original_path"]).parts.__len__(),
            d["modified_at"],
            d["original_path"],
        ),
    )


def decide_document(d: sqlite3.Row, settings: DecisionSettings, *, in_project: bool,
                    primary_id: int | None, override: sqlite3.Row | None,
                    similar_to: list[str] | None = None) -> Decision:
    payload = json.loads(d["classification_json"]) if d["classification_json"] else None
    c = payload["classification"] if payload else None
    ai_action = c["recommended_action"] if c else None
    decision = Decision(d["id"], LEAVE, None, ai_action=ai_action)

    def done(action: str, rule: str, reason: str, destination: str | None = "auto") -> Decision:
        decision.action = action
        decision.rules.append(rule)
        decision.reasons.append(reason)
        decision.destination = (
            destination_for(action, c["document_type"] if c else None) if destination == "auto" else destination
        )
        return decision

    if override is not None:
        return done(override["action"], "USER_OVERRIDE", "Your decision",
                    override["destination"] or "auto")

    if in_project:
        return done(LEAVE, "IN_PROJECT_FOLDER", "Part of a software project folder; moving it would break the project",
                    None)

    if primary_id is not None and primary_id != d["id"]:
        decision.duplicate_of = primary_id
        return done(CLEANUP, "DUPLICATE_COPY", f"Byte-identical copy of document #{primary_id}, which is kept",
                    DUPLICATES_DIR)

    if not d["supported"]:
        return done(LEAVE, "NOT_A_DOCUMENT", "Not a document type the organizer handles", None)

    is_image = d["extension"] in extractors.IMAGE_EXTENSIONS
    if is_image and d["extraction_status"] == extractors.EMPTY:
        return done(LEAVE, "PHOTO", "Photo or picture without document text", None)
    if is_image and d["extraction_status"] == extractors.NEEDS_OCR:
        return done(LEAVE, "IMAGE_NOT_READ", "Image; install Tesseract OCR to check it for document text", None)
    if d["extraction_status"] in extractors.REVIEW_STATUSES:
        from downloads_organizer.reporter import REVIEW_REASONS
        return done(REVIEW, "UNREADABLE", REVIEW_REASONS[d["extraction_status"]], REVIEW_DIR)

    if c is None:
        if d["classification_status"] == "invalid_output":
            return done(REVIEW, "INVALID_AI_OUTPUT", "The AI did not return a valid classification", REVIEW_DIR)
        return done(LEAVE, "NOT_CLASSIFIED", "Not classified yet; run 'classify'", None)

    doc_type, confidence = c["document_type"], c["confidence"]
    decision.reasons.append(f"AI: {c['summary']}")
    if similar_to:
        decision.rules.append("SIMILAR_VERSIONS")
        decision.reasons.append("Possibly another version of: " + ", ".join(similar_to[:3]))

    if confidence < settings.review_below_confidence:
        return done(REVIEW, "LOW_CONFIDENCE", f"AI confidence {confidence:.2f} is below "
                                              f"{settings.review_below_confidence}")
    if doc_type in settings.protected_types:
        reason = f"'{doc_type}' documents are always kept"
        if ai_action != KEEP:
            reason += f" (AI suggested {ai_action})"
        return done(KEEP, "PROTECTED_TYPE", reason)
    if ai_action == CLEANUP:
        if c["sensitivity"] in settings.protected_sensitivity:
            return done(REVIEW, "SENSITIVE_NOT_CLEANUP",
                        f"AI suggested cleanup, but the document looks {c['sensitivity']}; check it yourself")
        if confidence < settings.cleanup_min_confidence:
            return done(REVIEW, "UNSURE_CLEANUP", f"AI suggested cleanup with confidence {confidence:.2f}, "
                                                  f"below {settings.cleanup_min_confidence}")
        return done(CLEANUP, "AI_CLEANUP", "AI found no lasting value: " + "; ".join(c["reasons"][:2]))
    if ai_action in (REVIEW, "UNKNOWN") or doc_type == "Unknown":
        return done(REVIEW, "AI_UNSURE", "AI could not decide: " + "; ".join(c["reasons"][:2]))
    return done(ai_action, "AI_RECOMMENDATION", "; ".join(c["reasons"][:2]))


def decide_all(conn: sqlite3.Connection, config: Config) -> list[Decision]:
    """Recompute decisions for every present document and store them."""
    settings = config.decisions
    detector = ProjectDetector(config.downloads_dir, settings)
    docs = conn.execute("SELECT * FROM documents WHERE present = 1").fetchall()
    in_project = {d["id"]: detector.project_of(d["original_path"]) is not None for d in docs}
    by_id = {d["id"]: d for d in docs}

    primary: dict[int, int] = {}
    groups: dict[int, list[sqlite3.Row]] = {}
    for group_id, document_id in conn.execute(
        "SELECT group_id, document_id FROM duplicate_groups WHERE duplicate_type = 'exact'"
    ):
        if document_id in by_id:
            groups.setdefault(group_id, []).append(by_id[document_id])
    for members in groups.values():
        if len(members) > 1:
            keep = _primary_copy(members, in_project)
            for m in members:
                primary[m["id"]] = keep["id"]

    similar: dict[int, list[str]] = {}
    similar_groups: dict[int, list[int]] = {}
    for group_id, document_id in conn.execute(
        "SELECT group_id, document_id FROM duplicate_groups WHERE duplicate_type = 'similar'"
    ):
        if document_id in by_id:
            similar_groups.setdefault(group_id, []).append(document_id)
    for members in similar_groups.values():
        for m in members:
            others = {by_id[o]["filename"] for o in members if by_id[o]["sha256"] != by_id[m]["sha256"]}
            similar[m] = sorted(others)

    overrides = {r["document_id"]: r for r in conn.execute("SELECT * FROM user_overrides")}
    decisions = [
        decide_document(d, settings, in_project=in_project[d["id"]], primary_id=primary.get(d["id"]),
                        override=overrides.get(d["id"]), similar_to=similar.get(d["id"]))
        for d in docs
    ]

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with conn:
        conn.execute("DELETE FROM decisions")
        conn.executemany(
            "INSERT INTO decisions (document_id, action, destination, ai_action, rules, reasons, duplicate_of,"
            " decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(x.document_id, x.action, x.destination, x.ai_action, json.dumps(x.rules), json.dumps(x.reasons),
              x.duplicate_of, now) for x in decisions],
        )
    return decisions
