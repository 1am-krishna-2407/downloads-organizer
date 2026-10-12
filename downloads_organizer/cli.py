"""Command-line interface: ``downloads-organizer <command>`` (or ``python -m downloads_organizer``)."""

from __future__ import annotations

import argparse
import getpass
import logging
import logging.handlers
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

import yaml

from downloads_organizer import (
    __version__, classifier, database, decision_engine, extractors, organizer, pipeline, reporter,
)
from downloads_organizer.config import (
    PROVIDERS, classification_settings, load_config, system_downloads_dir, user_config_path,
)
from downloads_organizer.locking import AlreadyRunning, run_lock
from downloads_organizer.scanner import scan

PROG = "downloads-organizer"
TASK_NAME = "DownloadsOrganizer"


def _setup_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        logs_dir / "organizer.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise


def _load(args: argparse.Namespace, root: Path | None = None):
    config = load_config(getattr(args, "config", None), downloads_dir=root)
    _setup_logging(config.logs_dir)
    return config


def _progress(label: str, every: int = 1):
    def show(done: int, total: int) -> None:
        if done == total or done % every == 0:
            print(f"\r{label}: {done}/{total}", end="" if done < total else "\n", flush=True)
    return show


# --- read-only steps -------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    config = _load(args, args.root)
    print(f"Scanning (read-only): {config.downloads_dir}")
    result = scan(config)
    if not args.no_db:
        with closing(database.connect(config.database_path)) as conn:
            database.save_scan(conn, result)
    formats = ("csv", "xlsx") if args.format == "both" else (args.format,)
    reports = reporter.write_inventory(result, config.reports_dir, formats)

    summary = reporter.summarize(result)
    print(f"Files inventoried: {summary['files']} ({summary['supported_files']} supported)")
    print(f"Total size:        {summary['total_size']}")
    print(f"Errors:            {summary['errors']}")
    if summary["skipped"]:
        print("Skipped:           " + ", ".join(f"{k}={v}" for k, v in sorted(summary["skipped"].items())))
    for folder, kind in result.skipped_folders[:10]:
        print(f"  left alone ({kind}): {folder}")
    top = ", ".join(f"{ext} {n}" for ext, n in summary["by_extension"][:10])
    if top:
        print(f"Top extensions:    {top}")
    for path in reports:
        print(f"Report:            {path}")
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    config = _load(args, args.root)
    with run_lock(config.data_dir, "process"), closing(database.connect(config.database_path)) as conn:
        if not args.skip_scan:
            print(f"Scanning (read-only): {config.downloads_dir}")
            result = scan(config)
            database.save_scan(conn, result)
            print(f"Files found: {len(result.records)}")
        if not extractors.ocr_available(config.extraction):
            print("OCR: Tesseract not found; scanned PDFs go to review and images are left alone.")
        stats = pipeline.process_documents(conn, config, _progress("Reading files", every=10))
        reports = reporter.write_document_report(conn, config.reports_dir)

    print(f"Hashed:            {stats.hashed}")
    print(f"Text extracted:    {stats.extracted} (reused from cache: {stats.extraction_cache_hits})")
    print(f"Errors:            {stats.errors}")
    print(f"Duplicate groups:  {stats.duplicate_groups} ({stats.duplicate_files} files)")
    print(f"Similar versions:  {stats.similar_groups} group(s)")
    if stats.deferred:
        print(f"Waiting:           {stats.deferred} file(s) changed in the last few minutes; next run")
    print("Extraction status: " + ", ".join(f"{k}={v}" for k, v in stats.statuses.most_common()))
    for path in reports:
        print(f"Report:            {path}")
    return 0


# --- AI classification -------------------------------------------------------------------

# A made-up document, so checking the API never sends any of your files.
SAMPLE_DOCUMENT = classifier.DocumentInput(
    filename="sample_resume_2026.pdf",
    extension=".pdf",
    size_bytes=84_000,
    modified_at="2026-01-15T10:00:00+05:30",
    page_count=2,
    text=(
        "Alex Example\nSoftware Engineer\n\nEXPERIENCE\nBackend Developer, Example Corp (2023-2026)\n"
        "- Built Python data pipelines\n\nEDUCATION\nB.Tech Computer Science, Example University, 2023\n\n"
        "SKILLS\nPython, SQL, Docker"
    ),
)


def _classifier(config) -> classifier.DocumentClassifier:
    settings = config.classification
    client = classifier.NemotronClient(settings, classifier.load_api_key(settings))
    return classifier.DocumentClassifier(client, settings)


def check_api(config) -> bool:
    settings = config.classification
    doc_classifier = _classifier(config)
    print(f"Provider: {settings.provider}  ({settings.base_url})")
    try:
        for model in filter(None, dict.fromkeys([settings.model, settings.escalation_model])):
            print(f"Model: {model}  (sending a made-up sample resume)")
            try:
                call = doc_classifier.client.chat(model, classifier.build_messages(SAMPLE_DOCUMENT, settings))
                result = classifier.Classification.model_validate(classifier.parse_json_object(call.content))
            except classifier.ApiError as exc:
                print(f"  FAILED: {exc}")
                return False
            except Exception as exc:
                print(f"  The reply did not match the expected format: {exc}")
                return False
            print(f"  OK in {call.latency_ms} ms -> {result.document_type} / {result.recommended_action} "
                  f"(structured output: {doc_classifier.client.structured_mode})")
    finally:
        doc_classifier.client.close()
    return True


def cmd_check_api(args: argparse.Namespace) -> int:
    ok = check_api(_load(args))
    print("API check passed." if ok else "API check failed.")
    return 0 if ok else 1


def cmd_classify(args: argparse.Namespace) -> int:
    config = _load(args)
    settings = config.classification
    escalation = f" (second opinion: {settings.escalation_model})" if settings.escalation_model else ""
    print(f"Model: {settings.model}{escalation}")
    doc_classifier = None if args.dry_run else _classifier(config)
    with run_lock(config.data_dir, "classify"), closing(database.connect(config.database_path)) as conn:
        try:
            stats = pipeline.classify_documents(
                conn, config, doc_classifier, limit=args.limit, dry_run=args.dry_run,
                retry_failed=args.retry_failed, progress=_progress("Classifying"),
            )
        except classifier.ApiError as exc:
            print(f"\nStopped: {exc}", file=sys.stderr)
            print("Progress so far is saved; run again to continue.", file=sys.stderr)
            return 1
        finally:
            if doc_classifier:
                doc_classifier.client.close()

        if args.dry_run:
            print(f"Unique documents awaiting classification: {stats.candidates}")
            print(f"Would send now: {stats.classified} (~{stats.estimated_prompt_tokens:,} input tokens)")
            print("Nothing was sent.")
            return 0
        reports = reporter.write_document_report(conn, config.reports_dir)

    print(f"Classified:        {stats.classified} (reused from cache: {stats.cache_hits})")
    print(f"Second opinions:   {stats.escalated}")
    print(f"Invalid output:    {stats.invalid_output}")
    print(f"API errors:        {stats.errors}")
    if stats.api_calls:
        print(f"API calls:         {stats.api_calls}, {stats.prompt_tokens:,} input + "
              f"{stats.completion_tokens:,} output tokens, avg {stats.latency_ms // stats.api_calls} ms")
    if stats.actions:
        print("Suggestions:       " + ", ".join(f"{k}={v}" for k, v in stats.actions.most_common()))
    left = stats.candidates - stats.classified - stats.cache_hits - stats.invalid_output - stats.errors
    if left > 0:
        print(f"Not yet classified: {left} (run again to continue)")
    for path in reports:
        print(f"Report:            {path}")
    return 0


# --- decisions and moves -----------------------------------------------------------------


def _print_decisions(decisions) -> None:
    counts = Counter(d.action for d in decisions)
    print("Decisions:         " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))


def cmd_decide(args: argparse.Namespace) -> int:
    config = _load(args)
    with closing(database.connect(config.database_path)) as conn:
        decisions = decision_engine.decide_all(conn, config)
        report = reporter.write_review_report(conn, config.reports_dir)
    _print_decisions(decisions)
    print(f"Report:            {report}")
    return 0


def _confirm(prompt: str, word: str) -> bool:
    if not sys.stdin.isatty():
        print("Not running interactively; add --yes to confirm.", file=sys.stderr)
        return False
    return input(f"{prompt} Type {word} to continue: ").strip() == word


def cmd_organize(args: argparse.Namespace) -> int:
    config = _load(args)
    with closing(database.connect(config.database_path)) as conn:
        if args.history:
            rows = organizer.batches(conn)
            if not rows:
                print("No moves have been made yet.")
            for b in rows:
                print(f"{b['batch_id']}  moved={b['completed']}  failed={b['failed']}  "
                      f"rolled back={b['rolled_back']}  ({b['started']})")
            return 0

        if args.rollback:
            batch = args.rollback
            if batch == "last":
                rows = [b for b in organizer.batches(conn) if b["completed"]]
                if not rows:
                    print("Nothing to undo.")
                    return 0
                batch = rows[0]["batch_id"]
            if not args.yes and not _confirm(f"Move the files from batch {batch} back to where they were?", "UNDO"):
                return 1
            with run_lock(config.data_dir, "rollback"):
                restored, problems = organizer.rollback(conn, config, batch)
            print(f"Restored {restored} file(s) from batch {batch}.")
            for path, problem in problems:
                print(f"  Not restored: {path}: {problem}")
            return 0 if not problems else 1

        decision_engine.decide_all(conn, config)
        actions = tuple(a.strip().upper() for a in args.actions.split(","))
        moves, skipped = organizer.plan_moves(conn, config, actions)
        plan_path = reporter.write_move_plan(moves, skipped, config.reports_dir)

        print(f"Destination: {config.organized_dir}")
        print(f"Including:   {', '.join(actions)}")
        by_folder = Counter(m.destination.parent.relative_to(config.organized_dir).as_posix() for m in moves)
        for folder, n in sorted(by_folder.items()):
            print(f"  {n:5d}  {folder}")
        print(f"Planned moves: {len(moves)}   Skipped: {len(skipped)}")
        print(f"Full plan: {plan_path}")

        if not args.apply:
            print(f"Preview only; nothing was moved. Run '{PROG} organize --apply' to move these files,")
            print(f"or review and approve them in the dashboard ('{PROG} dashboard').")
            return 0
        if not moves:
            return 0
        if not args.yes and not _confirm(f"Move {len(moves)} file(s)? Nothing is deleted; this can be undone.",
                                         "MOVE"):
            print("Cancelled; nothing was moved.")
            return 1
        with run_lock(config.data_dir, "organize"):
            batch, moved, failures = organizer.apply_moves(conn, moves)
    print(f"Moved {moved} file(s) in batch {batch}. Undo with: {PROG} organize --rollback {batch}")
    for move, error in failures:
        print(f"  Failed: {move.source}: {error}")
    return 0 if not failures else 1


# --- automation ---------------------------------------------------------------------------


def run_pipeline(config, classify: bool = True) -> int:
    """Scan, process, classify and decide. Never moves files."""
    with run_lock(config.data_dir, "run"), closing(database.connect(config.database_path)) as conn:
        result = scan(config)
        database.save_scan(conn, result)
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] Scanned {len(result.records)} files in {config.downloads_dir}")
        stats = pipeline.process_documents(conn, config)
        if stats.deferred:
            print(f"  {stats.deferred} file(s) changed in the last few minutes; they'll be handled next run")
        if classify:
            try:
                doc_classifier = _classifier(config)
            except classifier.FatalApiError as exc:
                print(f"  Classification skipped: {exc}")
            else:
                try:
                    c = pipeline.classify_documents(conn, config, doc_classifier)
                    print(f"  Classified {c.classified} new document(s) ({c.api_calls} API calls)")
                except classifier.ApiError as exc:
                    print(f"  Classification stopped: {exc}")
                finally:
                    doc_classifier.client.close()
        decisions = decision_engine.decide_all(conn, config)
        reporter.write_document_report(conn, config.reports_dir)
        report = reporter.write_review_report(conn, config.reports_dir)
    _print_decisions(decisions)
    print(f"  Review: {PROG} dashboard   (report: {report})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return run_pipeline(_load(args), classify=not args.no_classify)


def cmd_watch(args: argparse.Namespace) -> int:
    config = _load(args)
    minutes = args.interval or config.automation.watch_interval_minutes
    print(f"Watching {config.downloads_dir} every {minutes} min (Ctrl+C to stop). Files are never moved automatically.")
    try:
        while True:
            try:
                run_pipeline(config)
            except AlreadyRunning as exc:
                print(f"  Skipped this round: {exc}")
            except Exception as exc:  # keep watching; the next round retries
                logging.getLogger(__name__).exception("Watch run failed")
                print(f"  Run failed: {exc}")
            time.sleep(minutes * 60)
    except KeyboardInterrupt:
        print("Stopped.")
        return 0


def _scheduled_python() -> str:
    """pythonw.exe on Windows, so scheduled runs don't flash a console window."""
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return str(windowless if sys.platform == "win32" and windowless.exists() else exe)


def cmd_schedule(args: argparse.Namespace) -> int:
    command = f'"{_scheduled_python()}" -m downloads_organizer run'
    if args.config:
        command += f' --config "{Path(args.config).resolve()}"'
    if sys.platform != "win32":
        hours = args.every_hours
        print("Add this line with 'crontab -e' to run automatically:")
        print(f"  0 */{hours} * * * {command} >> ~/.downloads-organizer-cron.log 2>&1")
        return 0

    if args.action == "status":
        result = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
                                capture_output=True, text=True)
        print(result.stdout.strip() if result.returncode == 0 else "Not scheduled.")
        return 0
    if args.action == "remove":
        result = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"], capture_output=True, text=True)
        print("Removed the scheduled run." if result.returncode == 0 else "Not scheduled.")
        return 0
    result = subprocess.run(
        ["schtasks", "/Create", "/TN", TASK_NAME, "/SC", "HOURLY", "/MO", str(args.every_hours),
         "/TR", command, "/F"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Could not create the scheduled task: {result.stderr.strip() or result.stdout.strip()}")
        return 1
    print(f"Scheduled: every {args.every_hours} hour(s), new downloads are read and classified.")
    print("Files are never moved automatically; review them in the dashboard.")
    print(f"Remove with: {PROG} schedule remove")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The dashboard needs Streamlit: pip install \"downloads-organizer[dashboard]\"", file=sys.stderr)
        return 1
    app_path = Path(__file__).resolve().parent / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.address", "localhost",
           "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"]
    if args.config:
        cmd += ["--", "--config", str(Path(args.config).resolve())]
    return subprocess.call(cmd)


# --- setup and diagnostics ---------------------------------------------------------------


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config) if args.config else user_config_path()
    interactive = sys.stdin.isatty() and not args.yes
    if path.exists() and not args.force:
        if not interactive or _ask(f"{path} already exists. Replace it? (y/N)", "n").lower() != "y":
            print(f"Keeping the existing config: {path}")
            return 0

    print("Downloads Organizer setup\n")
    detected = str(system_downloads_dir())
    downloads = args.downloads or (_ask("Folder to organize", detected) if interactive else detected)
    if not Path(downloads).expanduser().is_dir():
        print(f"Folder not found: {downloads}", file=sys.stderr)
        return 1

    provider = args.provider
    if not provider and interactive:
        print("\nWhich AI should read your documents?")
        print("  1. NVIDIA Nemotron, hosted (recommended; free key from build.nvidia.com)")
        print("  2. OpenRouter (one key for many providers)")
        print("  3. Ollama, on this computer (private; needs Ollama installed and ~3 GB of disk)")
        provider = {"1": "nvidia", "2": "openrouter", "3": "ollama"}.get(_ask("Choose 1-3", "1"), "nvidia")
    provider = provider or "nvidia"
    settings = classification_settings({"provider": provider})

    if settings.api_key_env and not settings.is_local:
        key = args.api_key
        if not key and interactive:
            print(f"\nGet a key at {settings.key_url} (it's stored only on this computer).")
            key = getpass.getpass("Paste your API key (hidden; Enter to skip): ").strip()
        if key:
            stored = classifier.save_api_key(settings.api_key_env, key)
            print(f"Key saved to {stored}")
        else:
            print(f"No key entered. Add one later with '{PROG} init', or set {settings.api_key_env}.")
    elif provider == "ollama":
        print(f"\nInstall Ollama from https://ollama.com, then run: ollama pull {settings.model}")

    content: dict = {"classification": {"provider": provider}}
    if Path(downloads).expanduser().resolve() != Path(detected).resolve():
        content["downloads_dir"] = str(Path(downloads).expanduser().resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Downloads Organizer settings. Only what differs from the defaults needs to be here;\n"
        f"# see {Path(__file__).resolve().parent / 'default_config.yaml'} for every option.\n"
        + yaml.safe_dump(content, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nSettings saved to {path}")

    config = load_config(path)
    if interactive and _ask("Test the AI connection now with a made-up document? (Y/n)", "y").lower() == "y":
        try:
            check_api(config)
        except classifier.FatalApiError as exc:
            print(f"  {exc}")
    print(f"\nNext: '{PROG} run' to analyse {config.downloads_dir}, then '{PROG} dashboard' to review.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(getattr(args, "config", None))
    problems = 0

    def line(ok: bool | None, label: str, detail: str) -> None:
        nonlocal problems
        mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
        problems += ok is False
        print(f"[{mark}] {label:<18} {detail}")

    print(f"{PROG} {__version__}  (Python {platform.python_version()}, {platform.system()} {platform.release()})\n")
    line(True if config.source else None, "Config", str(config.source or f"defaults (run '{PROG} init')"))
    line(config.downloads_dir.is_dir(), "Downloads folder", str(config.downloads_dir))
    line(None, "Organized into", str(config.organized_dir))
    try:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        probe = config.data_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        line(True, "Data folder", str(config.data_dir))
    except OSError as exc:
        line(False, "Data folder", f"{config.data_dir} is not writable: {exc}")
    if config.database_path.exists():
        with closing(database.connect(config.database_path)) as conn:
            files, classified = conn.execute(
                "SELECT COUNT(*), SUM(classification_status = 'done') FROM documents WHERE present = 1").fetchone()
            last = conn.execute("SELECT MAX(finished_at) FROM scan_runs").fetchone()[0]
        line(True, "Database", f"{files} files tracked, {classified or 0} classified, last scan {last or 'never'}")
    else:
        line(None, "Database", f"not created yet (run '{PROG} run')")

    tessdata = extractors.tessdata_path(config.extraction.tessdata_dir)
    if tessdata:
        langs = sorted(p.stem for p in Path(tessdata).glob("*.traineddata"))
        line(True, "OCR (Tesseract)", f"{tessdata}  languages: {', '.join(langs)}")
    else:
        line(None, "OCR (Tesseract)", "not found; scanned PDFs can't be read. Install: "
             + ("winget install UB-Mannheim.TesseractOCR" if sys.platform == "win32"
                else "brew install tesseract" if sys.platform == "darwin" else "sudo apt install tesseract-ocr"))

    s = config.classification
    line(True, "AI provider", f"{s.provider}: {s.model}" + (f" (+ {s.escalation_model})" if s.escalation_model else ""))
    try:
        key = classifier.load_api_key(s)
        line(True, "API key", "not needed (local model)" if not key else f"found ({key[:6]}...{key[-4:]})")
    except classifier.FatalApiError as exc:
        line(False, "API key", str(exc))
    try:
        import streamlit  # noqa: F401
        line(True, "Dashboard", "available")
    except ImportError:
        line(None, "Dashboard", "not installed: pip install \"downloads-organizer[dashboard]\"")
    if sys.platform == "win32":
        scheduled = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME], capture_output=True).returncode == 0
        line(None, "Scheduled runs", "on" if scheduled else f"off ('{PROG} schedule install' to turn on)")

    print("\nAll good." if not problems else f"\n{problems} problem(s) found.")
    return 0 if not problems else 1


# --- argument parsing -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Organize your Downloads folder with AI, safely: nothing moves without your approval "
                    "and nothing is ever deleted.",
        epilog=f"Getting started: {PROG} init, then {PROG} run, then {PROG} dashboard.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--config", type=Path, help="settings file (default: your per-user config)")
        p.set_defaults(func=func)
        return p

    p = add("init", cmd_init, "first-time setup: folder, AI provider and API key")
    p.add_argument("--downloads", help="folder to organize (default: your Downloads folder)")
    p.add_argument("--provider", choices=sorted(PROVIDERS), help="AI provider")
    p.add_argument("--api-key", help="API key (prefer the hidden prompt or an environment variable)")
    p.add_argument("--yes", action="store_true", help="no questions; use defaults")
    p.add_argument("--force", action="store_true", help="replace an existing config")

    add("doctor", cmd_doctor, "check the setup: folders, OCR, AI provider, API key")

    p = add("run", cmd_run, "scan, read, classify and decide in one go (never moves files)")
    p.add_argument("--no-classify", action="store_true", help="skip the AI step")

    add("dashboard", cmd_dashboard, "open the review dashboard in your browser")

    p = add("organize", cmd_organize, "preview, apply or undo moves (preview by default)")
    p.add_argument("--actions", default="KEEP,ARCHIVE",
                   help="decisions to carry out: KEEP, ARCHIVE, REVIEW, CLEANUP_CANDIDATE (default KEEP,ARCHIVE)")
    p.add_argument("--apply", action="store_true", help="actually move the files (asks for confirmation)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--rollback", metavar="BATCH", help="move a batch back; 'last' for the most recent")
    p.add_argument("--history", action="store_true", help="list past move batches")

    p = add("schedule", cmd_schedule, "run automatically in the background (never moves files)")
    p.add_argument("action", nargs="?", choices=["install", "remove", "status"], default="install")
    p.add_argument("--every-hours", type=int, default=1, help="how often (default 1)")

    p = add("watch", cmd_watch, "keep running in this window, checking for new downloads")
    p.add_argument("--interval", type=int, help="minutes between checks (default from config)")

    p = add("scan", cmd_scan, "step 1: list files (read-only)")
    p.add_argument("--root", type=Path, help="folder to scan instead of the configured one")
    p.add_argument("--format", choices=["csv", "xlsx", "both"], default="both")
    p.add_argument("--no-db", action="store_true", help="don't record the scan")

    p = add("process", cmd_process, "step 2: read text, find duplicates (read-only)")
    p.add_argument("--root", type=Path, help="folder to scan instead of the configured one")
    p.add_argument("--skip-scan", action="store_true", help="use the last scan")

    add("check-api", cmd_check_api, "test the AI provider with a made-up document")

    p = add("classify", cmd_classify, "step 3: classify documents with AI")
    p.add_argument("--limit", type=int, help="at most N new documents (try a small sample first)")
    p.add_argument("--dry-run", action="store_true", help="show what would be sent, without sending")
    p.add_argument("--retry-failed", action="store_true", help="also retry documents with invalid AI output")

    add("decide", cmd_decide, "step 4: apply the safety rules and write the review report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError, classifier.FatalApiError, AlreadyRunning) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Progress so far is saved.", file=sys.stderr)
        return 130
