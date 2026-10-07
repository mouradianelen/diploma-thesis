"""Download the DLPFC .h5ad sections from figshare into data/.

Queries the figshare API for article 29146307, downloads every file into data/,
verifies each MD5 checksum, and skips files that are already present and valid.

Usage:
    python scripts/download_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

ARTICLE_API = "https://api.figshare.com/v2/articles/29146307"
DATA_DIR = Path("data")


def md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import json

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(ARTICLE_API) as resp:
        files = json.load(resp)["files"]

    print(f"Found {len(files)} files on figshare.")

    for entry in files:
        name = entry["name"]
        expected = entry.get("supplied_md5")
        dest = DATA_DIR / name

        if dest.exists() and expected and md5(dest) == expected:
            print(f"  {name}: already present, skipping")
            continue

        size_mb = entry["size"] / 1e6
        print(f"  {name}: downloading ({size_mb:.0f} MB) ...", flush=True)
        urllib.request.urlretrieve(entry["download_url"], dest)

        if expected and md5(dest) != expected:
            print(f"  {name}: MD5 mismatch, download may be corrupt", file=sys.stderr)
            return 1

    print("Done. All sections are in data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
