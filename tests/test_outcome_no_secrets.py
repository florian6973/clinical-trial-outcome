"""Repository-level regression guard against committed model-provider secrets."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "configs", "paper", "tests", "README.md", "pyproject.toml")
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# Assemble provider prefixes so this test's own source does not contain a
# credential-shaped literal.  Long random-looking values are required; normal
# identifiers such as HF_TOKEN, OPENAI_API_KEY, and hf_device_map do not match.
TOKEN_PATTERNS = {
    "Hugging Face": re.compile(r"\b" + "hf" + r"_[A-Za-z0-9]{20,}\b"),
    "OpenAI": re.compile(r"\b" + "sk" + r"-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
}


def repository_text_files() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *SCAN_ROOTS,
    ]
    output = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    paths: list[Path] = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8"))
        if "archive" in relative.parts or "__pycache__" in relative.parts:
            continue
        path = ROOT / relative
        if path.is_file() and (path.suffix.casefold() in TEXT_SUFFIXES or path.name == "README.md"):
            paths.append(path)
    return sorted(set(paths))


def test_outcome_active_repository_contains_no_provider_secrets() -> None:
    violations: list[str] = []
    for path in repository_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for provider, pattern in TOKEN_PATTERNS.items():
            for line_number, line in enumerate(text.splitlines(), start=1):
                match = pattern.search(line)
                if match:
                    token = match.group(0)
                    redacted = f"{token[:5]}...{token[-4:]}"
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{line_number}: {provider} token {redacted}")
    assert not violations, "Credential-shaped literals found:\n" + "\n".join(violations)


def test_outcome_secret_patterns_ignore_environment_variable_names() -> None:
    safe_identifiers = "HF_TOKEN OPENAI_API_KEY hf_device_map huggingface_token"
    assert all(pattern.search(safe_identifiers) is None for pattern in TOKEN_PATTERNS.values())
