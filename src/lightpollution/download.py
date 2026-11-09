"""Download a pinned annual Black Marble release without exposing credentials."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

HOST = "ladsweb.modaps.eosdis.nasa.gov"
BASE = f"https://{HOST}"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
TILES = ("h20v02", "h21v02", "h20v03", "h21v03")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_from_environment() -> str:
    """A local file is an explicit alternative to BLACKMARBLE_TOKEN."""
    token = os.environ.get("BLACKMARBLE_TOKEN", "").strip()
    if not token and os.environ.get("BLACKMARBLE_TOKEN_FILE"):
        token = Path(os.environ["BLACKMARBLE_TOKEN_FILE"]).read_text().strip()
    if not token:
        raise ValueError("Set BLACKMARBLE_TOKEN or BLACKMARBLE_TOKEN_FILE.")
    return token


def select_files(entries: list[dict], year: int, tiles: tuple[str, ...]) -> list[dict]:
    """Prefer the latest processing timestamp; never guess a filename."""
    selected = []
    for tile in tiles:
        pattern = re.compile(rf"^VNP46A4\.A{year}001\.{tile}\.002\.\d{{13}}\.h5$")
        candidates = [e for e in entries if pattern.fullmatch(e.get("name", ""))]
        if not candidates:
            raise ValueError(f"No VNP46A4.002 file for {year}, {tile}.")
        selected.append(max(candidates, key=lambda e: e["name"]))
    return selected


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != HOST or parsed.username:
        raise ValueError("Refusing to send the NASA token outside the LAADS HTTPS host.")


def verify_file(path: Path, expected_size: int, expected_hash: str | None = None) -> str:
    if path.stat().st_size != expected_size:
        raise ValueError(f"Wrong byte count for {path.name}.")
    with path.open("rb") as stream:
        if stream.read(8) != HDF5_MAGIC:
            raise ValueError(f"{path.name} is not HDF5 (possibly an HTML login page).")
    digest = sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path.name}.")
    return digest


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def catalogue(year: int = 2025, tiles: tuple[str, ...] = TILES) -> dict:
    listing = f"{BASE}/api/v2/content/details/allData/5200/VNP46A4/{year}/001"
    response = requests.get(listing, timeout=(20, 60))
    response.raise_for_status()
    records = select_files(response.json()["content"], year, tiles)
    return {
        "product": "VNP46A4", "collection": "002", "archive_set": "5200",
        "year": year, "tiles": list(tiles), "listing_url": listing,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [{**record,"download_verified":False} for record in records],
    }


def download_annual(
    directory: Path,
    year: int = 2025,
    tiles: tuple[str, ...] = TILES,
    manifest_path: Path | None = None,
) -> dict:
    """A saved manifest pins filenames and hashes for subsequent runs."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["year"] != year or manifest["tiles"] != list(tiles):
            raise ValueError("Manifest year/tiles differ from the requested release.")
        records = manifest["files"]
    else:
        manifest = catalogue(year,tiles)
        records = manifest["files"]
    # Pin the selected release even if authentication or a later tile fails.
    write_manifest(manifest_path,manifest)
    token = None
    for record in records:
        name = record["name"]
        if Path(name).name != name:
            raise ValueError("Invalid filename in NASA manifest.")
        target = directory / name
        size = int(record["size"])
        if target.exists():
            record["sha256"] = verify_file(target, size, record.get("sha256"))
            record["download_verified"] = True
            write_manifest(manifest_path,manifest)
            print(f"Verified cached {name}", flush=True)
            continue
        if token is None:
            token = token_from_environment()
        url = record["downloadsLink"]
        validate_url(url)
        temporary = target.with_suffix(".h5.part")
        print(f"Downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        try:
            with requests.get(
                url, headers={"Authorization": f"Bearer {token}"},
                stream=True, timeout=(20, 120), allow_redirects=False,
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308, 401, 403):
                    location = urlparse(response.headers.get("Location","")).path
                    explanation = " Complete the LAADS web sign-in and any license/profile requirements." if "/profiles/licenses/" in location else " Check Earthdata/LAADS authorization."
                    raise RuntimeError(f"NASA download not authorized (HTTP {response.status_code})."+explanation)
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            record["sha256"] = verify_file(temporary, size, record.get("sha256"))
            temporary.replace(target)
            record["download_verified"] = True
            write_manifest(manifest_path,manifest)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}", flush=True)
    write_manifest(manifest_path,manifest)
    return manifest
