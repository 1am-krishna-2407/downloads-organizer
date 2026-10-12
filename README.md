# Downloads Organizer

**Your Downloads folder, sorted by what's actually inside each file.** Resumes, payslips, certificates, receipts, notes, e-books: Downloads Organizer reads them (including scans and photos of documents), asks an AI what each one is, and proposes a tidy folder for it. Nothing moves until you approve it, nothing is ever deleted, and every move can be undone.

```
Downloads ─► list files ─► read text (+OCR) ─► spot duplicates & versions ─► AI classifies ─► safety rules decide
          ─► you review in the dashboard ─► approve ─► verified move ─► undo any time
```

## What it does

- **Reads your documents:** PDF, Word, PowerPoint, Excel, text, RTF, and images or scanned PDFs through OCR.
- **Finds exact duplicates:** byte-identical copies such as `report (1).pdf`. It keeps one and flags the rest.
- **Spots versions of the same document:** drafts, re-exports and tailored resumes.
- **Classifies each document** as a resume, payslip, certificate, invoice, notes, reference book and so on, with a reason for each.
- **Follows safety rules the AI can't override.** For example, certificates, offer letters, financial and legal documents are always kept.
- **Leaves alone what isn't a personal document:** photos, videos, installers, unzipped apps, code projects and browser extensions.
- **Shows everything in a local dashboard** where you review, change decisions and approve moves.
- **Can run on a schedule**, picking up new downloads and waiting for your approval.

## Install

Requires Python 3.10+.

```bash
pip install "downloads-organizer[dashboard] @ git+https://github.com/<you>/downloads-organizer"
# or, from a checkout of this repository:
pip install -e ".[dashboard]"
```

**OCR (recommended)** lets it read scanned PDFs and photos of documents:

| OS | Command |
|---|---|
| Windows | `winget install UB-Mannheim.TesseractOCR` |
| macOS | `brew install tesseract` |
| Linux | `sudo apt install tesseract-ocr` |

## Get started

```bash
downloads-organizer init        # pick the folder and an AI provider, enter a key (stored only on your PC)
downloads-organizer run         # analyse everything; nothing is moved
downloads-organizer dashboard   # review, adjust, approve
```

Check your setup at any time with `downloads-organizer doctor`.

### Choosing an AI provider

| Provider | Privacy | Cost | Setup |
|---|---|---|---|
| **NVIDIA Nemotron** (default) | Excerpts sent to NVIDIA, with personal numbers masked | Free tier is enough for most folders | Key from [build.nvidia.com](https://build.nvidia.com) |
| **OpenRouter** | Excerpts sent to OpenRouter | Pay as you go | Key from [openrouter.ai/keys](https://openrouter.ai/keys) |
| **Ollama** (local) | **Nothing leaves your computer** | Free | Install [Ollama](https://ollama.com), then `ollama pull nemotron-3-nano:4b` |

Any OpenAI-compatible endpoint also works with `provider: custom` plus `base_url` and `model`.

With NVIDIA, documents are classified by **Nemotron 3.5 Lightning**. When it isn't confident, **Nemotron 3 Super** gives a second opinion.

## Everyday use

| Command | What it does |
|---|---|
| `run` | Scans, reads, classifies and decides. **Never moves files.** |
| `dashboard` | Review queue, all files, duplicates and versions, move files, history and undo |
| `organize` | Previews moves in the terminal. Add `--apply` to move after you type `MOVE` to confirm, and `--rollback last` to undo. |
| `schedule install --every-hours 4` | Runs automatically in the background. Windows uses Task Scheduler; macOS and Linux get a cron line to add. |
| `watch` | Keeps running in a terminal window and checks every 30 minutes |
| `doctor` | Checks folders, OCR, provider, API key and dashboard |

Advanced users can run each step on its own: `scan`, `process`, `classify` (with `--dry-run` and `--limit N`), then `decide`.

## Where files go

```
Downloads/Organized/
├── Career/{Resumes, Job_Descriptions, Offers}
├── Education/{Notes, Assignments, Certificates}
├── Finance/{Receipts}   Legal_and_Official/   Projects/   Reference/   Miscellaneous/
├── Archive/<same folders>          ← useful, but not needed day to day
├── Needs_Review/
└── Cleanup_Candidates/{Duplicates} ← you decide whether to delete these yourself
```

## How decisions are made

The AI only *suggests*. These rules, applied in order, make the final decision:

1. **Your own decision** from the dashboard always wins.
2. Files inside **code projects or unpacked apps** stay where they are, because moving them would break them.
3. **Extra byte-identical copies** become cleanup candidates. One copy is always kept.
4. **Photos without document text, videos and installers** stay where they are.
5. **Unreadable documents** (encrypted, corrupt, empty) go to review.
6. **Low AI confidence** sends the file to review.
7. **Resumes, certificates, offer letters, financial, government and legal documents** are always kept.
8. **Cleanup suggestions** for sensitive documents, or made with low confidence, go to review.
9. Otherwise, the AI's suggestion is used.

## Safety and privacy

- **No deleting.** Cleanup candidates are only gathered in one place for you to look at.
- **No overwriting.** Name clashes get `(2)` added. Undo won't replace a file that has since appeared at the original path.
- **Verified moves.** Each move is logged first and confirmed afterwards. Moves to another drive copy the file, compare hashes, then remove the original.
- **Changed files are skipped.** A file that changed after it was analysed is skipped. Files still downloading wait for the next run.
- **One run at a time.** Scheduled runs, `watch`, the CLI and the dashboard can't collide.
- **Minimal data leaves your PC:** only a short excerpt per document, with emails, phone numbers, card, bank, Aadhaar, PAN and passport numbers masked first. Photos are never sent. With Ollama, nothing is sent at all.
- **Prompt-injection safe.** Document text is treated as data, never as instructions, and the rules above always have the final say.
- **Your data stays local.** It's kept in one folder, `~/.downloads-organizer` (Windows) or the standard app-data folder (macOS/Linux): settings, API key, database, reports and logs. Document text is never written to logs.

## Configuration

`downloads-organizer init` creates a small `config.yaml`, and you only add what you want to change. Every option, with explanations, is listed in [`downloads_organizer/default_config.yaml`](downloads_organizer/default_config.yaml). Common changes:

```yaml
downloads_dir: D:/Downloads
organized_dir: D:/Sorted
exclude_dirs: [Torrents]
extraction:
  ocr_language: eng+hin        # needs the Hindi language pack for Tesseract
classification:
  provider: ollama
decisions:
  protected_types: [Resume or CV, Certificate, Financial document]
```

## Development

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The tests need no API key and no network. The OCR tests run when Tesseract is installed. CI runs on Windows, macOS and Linux with Python 3.10 and 3.13.

```text
downloads_organizer/
  cli.py               commands
  config.py            settings, per-user folders, provider presets
  default_config.yaml  every option, documented
  scanner.py           read-only listing; skips software and code folders
  extractors.py        PDF/Office/text/OCR text extraction
  hashing.py           exact duplicates
  similarity.py        near-duplicates (MinHash + Jaccard)
  classifier.py        AI client, prompt, schema, masking
  pipeline.py          read → dedupe → classify, resumable, parallel
  decision_engine.py   safety rules
  organizer.py         planned, verified, reversible moves
  dashboard.py         Streamlit review UI
  reporter.py          Excel/CSV reports
  database.py          SQLite (WAL) schema and migrations
  locking.py           one run at a time
```
