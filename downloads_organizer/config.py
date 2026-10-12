"""Settings: packaged defaults, overridden by the user's config file.

Lookup order for the config file: explicit path > $SDO_CONFIG > the per-user
config folder. Generated data lives in the per-user data folder unless the
config says otherwise. Set $SDO_HOME to keep everything in one folder instead.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path

import yaml

APP_NAME = "DownloadsOrganizer"
APP_SLUG = "downloads-organizer"
PACKAGE_DIR = Path(__file__).resolve().parent


# --- per-user locations ------------------------------------------------------------


def _home_override() -> Path | None:
    home = os.environ.get("SDO_HOME")
    return Path(home).expanduser().resolve() if home else None


# On Windows everything lives in one folder in the user profile, like ~/.ollama or ~/.aws.
# AppData is avoided on purpose: Windows silently redirects AppData writes made by
# packaged apps (Microsoft Store Python, MSIX terminals), so the same user could end
# up with two different, invisible copies of their settings and history.
WINDOWS_HOME = Path.home() / f".{APP_SLUG}"


def user_config_dir() -> Path:
    if home := _home_override():
        return home
    if sys.platform == "win32":
        return WINDOWS_HOME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_SLUG


def user_data_dir() -> Path:
    if home := _home_override():
        return home
    if sys.platform == "win32":
        return WINDOWS_HOME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_SLUG


def user_config_path() -> Path:
    return user_config_dir() / "config.yaml"


def system_downloads_dir() -> Path:
    """The OS Downloads folder, honouring a relocated folder on Windows and XDG on Linux."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
            )
            with key:
                value, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
            path = Path(os.path.expandvars(value))
            if path.is_dir():
                return path
        except OSError:
            pass
    elif sys.platform != "darwin":
        dirs_file = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "user-dirs.dirs"
        try:
            for line in dirs_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("XDG_DOWNLOAD_DIR="):
                    value = line.split("=", 1)[1].strip().strip('"').replace("$HOME", str(Path.home()))
                    if Path(value).is_dir():
                        return Path(value)
        except OSError:
            pass
    return Path.home() / "Downloads"


# --- settings ------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionSettings:
    max_file_mb: int = 100
    max_pages: int = 50
    max_chars: int = 100_000
    # A PDF averaging fewer characters per page than this is treated as scanned.
    min_chars_per_page: int = 25
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_max_pages: int = 10
    # An OCR'd image with fewer real words than this is a photo, not a document.
    min_ocr_words: int = 8
    # Folder containing Tesseract's language data; None = auto-detect.
    tessdata_dir: str | None = None
    # Parallel extraction processes; 0 = automatic (up to 4, leaving one CPU core free).
    workers: int = 0


# Provider presets. Anything set explicitly in the config overrides these.
PROVIDERS: dict[str, dict] = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "escalation_model": "nvidia/nemotron-3-super-120b-a12b",
        "api_key_env": "NVIDIA_API_KEY",
        "key_url": "https://build.nvidia.com",
        "requests_per_minute": 30,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-3.5-lightning",
        "escalation_model": None,
        "api_key_env": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "requests_per_minute": 30,
    },
    "ollama": {  # runs on this computer: nothing leaves it, no key needed
        "base_url": "http://localhost:11434/v1",
        "model": "nemotron-3-nano:4b",
        "escalation_model": None,
        "api_key_env": None,
        "key_url": None,
        "requests_per_minute": 0,
        "concurrency": 1,
        "timeout_seconds": 300,
        "redact_personal_data": False,
    },
    "custom": {
        "base_url": None,
        "model": None,
        "escalation_model": None,
        "api_key_env": "SDO_API_KEY",
        "key_url": None,
        "requests_per_minute": 30,
    },
}


@dataclass(frozen=True)
class ClassificationSettings:
    provider: str = "nvidia"
    base_url: str = PROVIDERS["nvidia"]["base_url"]
    model: str = PROVIDERS["nvidia"]["model"]
    # Second opinion for low-confidence results; None disables escalation.
    escalation_model: str | None = PROVIDERS["nvidia"]["escalation_model"]
    api_key_env: str | None = "NVIDIA_API_KEY"
    key_url: str | None = "https://build.nvidia.com"
    escalate_below_confidence: float = 0.6
    # Text sent per document: the start, plus the end of longer documents.
    head_chars: int = 4000
    tail_chars: int = 1500
    redact_personal_data: bool = True
    temperature: float = 0.1
    max_output_tokens: int = 1024
    disable_thinking: bool = True
    # How to request schema-constrained JSON: "response_format" (OpenAI json_schema),
    # "nvext" (NIM guided_json) or "none". Unsupported options are dropped automatically;
    # output is always validated locally.
    structured_output: str = "response_format"
    timeout_seconds: float = 60
    max_retries: int = 4
    requests_per_minute: int = 30
    concurrency: int = 4

    @property
    def is_local(self) -> bool:
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        return host in ("localhost", "127.0.0.1", "[::1]", "::1")


@dataclass(frozen=True)
class DecisionSettings:
    review_below_confidence: float = 0.6
    cleanup_min_confidence: float = 0.8
    # Always KEEP these, whatever the AI suggests.
    protected_types: tuple[str, ...] = (
        "Resume or CV",
        "Certificate",
        "Offer letter or employment document",
        "Financial document",
        "Government or legal document",
    )
    # Never a cleanup candidate when the AI flags this sensitivity.
    protected_sensitivity: tuple[str, ...] = ("financial", "identity", "legal", "medical")
    # A folder containing any of these is a software project; its files are never moved.
    project_markers: tuple[str, ...] = (
        ".git", "package.json", "pyproject.toml", "setup.py", "requirements.txt",
        "CMakeLists.txt", "Makefile", "pom.xml", "build.gradle", "Cargo.toml", "go.mod", ".sln", ".csproj",
        "manifest.json",
    )
    project_name_patterns: tuple[str, ...] = (r"-(master|main)$",)
    similar_threshold: float = 0.8


@dataclass(frozen=True)
class AutomationSettings:
    # Files modified more recently than this may still be downloading; they wait for the next run.
    min_file_age_seconds: int = 120
    watch_interval_minutes: int = 30


@dataclass(frozen=True)
class Config:
    downloads_dir: Path
    organized_dir: Path
    data_dir: Path
    reports_dir: Path
    logs_dir: Path
    recursive: bool = True
    exclude_dirs: tuple[Path, ...] = ()
    skip_software_folders: bool = True
    skip_project_folders: bool = True
    supported_extensions: frozenset[str] = frozenset()
    skip_extensions: frozenset[str] = frozenset()
    skip_names: frozenset[str] = frozenset()
    skip_name_prefixes: tuple[str, ...] = ()
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    classification: ClassificationSettings = field(default_factory=ClassificationSettings)
    decisions: DecisionSettings = field(default_factory=DecisionSettings)
    automation: AutomationSettings = field(default_factory=AutomationSettings)
    source: Path | None = None  # the user config file in use, if any

    @property
    def database_path(self) -> Path:
        return self.data_dir / "organizer.db"

    def excluded_dirs(self) -> tuple[Path, ...]:
        """Folders the scanner must never descend into, including its own output."""
        return (
            self.organized_dir,
            self.data_dir,
            self.reports_dir,
            self.logs_dir,
            *self.exclude_dirs,
        )


# --- loading -------------------------------------------------------------------------


def _expand(value: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve(value: str | os.PathLike, base: Path) -> Path:
    path = _expand(value)
    return (path if path.is_absolute() else base / path).resolve()


def _norm_ext(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def default_settings() -> dict:
    text = resources.files(__package__).joinpath("default_config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def find_config_file(path: str | os.PathLike | None = None) -> Path | None:
    if path:
        return Path(path)
    if env := os.environ.get("SDO_CONFIG"):
        return Path(env)
    candidate = user_config_path()
    return candidate if candidate.exists() else None


def load_config(
    path: str | os.PathLike | None = None,
    downloads_dir: str | os.PathLike | None = None,
) -> Config:
    """Packaged defaults merged with the user's config. ``downloads_dir`` overrides the scan root."""
    config_path = find_config_file(path)
    user: dict = {}
    if config_path is not None:
        with open(config_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    raw = _merge(default_settings(), user)
    unknown = set(raw) - {f.name for f in fields(Config)} - {"classification", "extraction", "decisions",
                                                             "automation"}
    if unknown:
        raise ValueError(f"Unknown settings in {config_path}: {', '.join(sorted(unknown))}")

    base = config_path.resolve().parent if config_path else Path.cwd()
    data_home = user_data_dir()

    def path_setting(key: str, default: Path) -> Path:
        value = raw.get(key)
        return _resolve(value, base) if value else default.resolve()

    downloads = (
        _expand(downloads_dir).resolve() if downloads_dir else path_setting("downloads_dir", system_downloads_dir())
    )
    data_dir = path_setting("data_dir", data_home / "data")

    return Config(
        downloads_dir=downloads,
        organized_dir=path_setting("organized_dir", downloads / "Organized"),
        data_dir=data_dir,
        reports_dir=path_setting("reports_dir", data_home / "reports"),
        logs_dir=path_setting("logs_dir", data_home / "logs"),
        recursive=bool(raw.get("recursive", True)),
        exclude_dirs=tuple(_resolve(d, downloads) for d in raw.get("exclude_dirs") or []),
        skip_software_folders=bool(raw.get("skip_software_folders", True)),
        skip_project_folders=bool(raw.get("skip_project_folders", True)),
        supported_extensions=frozenset(_norm_ext(e) for e in raw.get("supported_extensions") or []),
        skip_extensions=frozenset(_norm_ext(e) for e in raw.get("skip_extensions") or []),
        skip_names=frozenset(n.lower() for n in raw.get("skip_names") or []),
        skip_name_prefixes=tuple(p.lower() for p in raw.get("skip_name_prefixes") or []),
        extraction=_section(ExtractionSettings, "extraction", raw),
        classification=classification_settings(raw.get("classification") or {}),
        decisions=_section(DecisionSettings, "decisions", raw),
        automation=_section(AutomationSettings, "automation", raw),
        source=config_path,
    )


def classification_settings(values: dict) -> ClassificationSettings:
    """Apply the provider preset, then any explicit settings."""
    provider = values.get("provider", "nvidia")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    merged = {**PROVIDERS[provider], **{k: v for k, v in values.items() if v is not None or k == "escalation_model"}}
    if not merged.get("base_url") or not merged.get("model"):
        raise ValueError("The 'custom' provider needs classification.base_url and classification.model")
    return _section(ClassificationSettings, "classification", {"classification": merged})


def _section(cls, name: str, raw: dict):
    values = raw.get(name) or {}
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
    # YAML lists become tuples so settings stay immutable.
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})
