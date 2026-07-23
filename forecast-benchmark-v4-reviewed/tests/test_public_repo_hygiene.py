from pathlib import Path

# Build these strings indirectly so this test does not trip on its own source.
BLOCKLIST = [
    "D" + "DE",
    "Env" + "erus",
    "Land" + "trac",
    "AWS_SECRET" + "_ACCESS_KEY",
    "SUPABASE_SERVICE" + "_ROLE_KEY",
    "postgres" + "ql://",
    "/home" + "/ubuntu",
]

TEXT_EXTENSIONS = {".md", ".py", ".toml", ".yml", ".yaml", ".txt"}


def test_public_repo_does_not_contain_private_boundary_terms():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        if ".git" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        for term in BLOCKLIST:
            if term in text:
                offenders.append(f"{path.relative_to(root)} contains blocked boundary term")
    assert offenders == []
