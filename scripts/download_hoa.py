"""Resumable parallel download of a full HOA N5 dataset from the public GCS
bucket. Uses plain HTTP with per-request retries so it survives flaky
networks that reset connections (the ESRF/GCS endpoints do on this link).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

BUCKET = "ucl-hip-ct-35a68e99feaae8932b1d44da0358940b"
PREFIX = "LADAF-2020-27/lung-left/2.45um_VOI-06-lower-lobe-basal_bm05/"
DEST = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "LADAF-2020-27_lung_left_2.45um_VOI-06-lower-lobe-basal_bm05"
)
THREADS = 16
API = "https://storage.googleapis.com/storage/v1/b"

session = requests.Session()


def _retry(fn, attempts: int = 12, base_sleep: float = 1.5):
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(base_sleep * 2**attempt, 45))
    raise last  # type: ignore[misc]


def list_objects() -> list[tuple[str, int]]:
    names: list[tuple[str, int]] = []
    token = None
    page = 0
    while True:
        params = {"prefix": PREFIX, "delimiter": "", "maxResults": 1000}
        if token:
            params["pageToken"] = token

        def fetch_page(params=params):
            r = session.get(f"{API}/{BUCKET}/o", params=params, timeout=(20, 120))
            r.raise_for_status()
            return r.json()

        js = _retry(fetch_page)
        for item in js.get("items", []):
            if "size" in item:  # skip folder placeholder objects
                names.append((item["name"], int(item["size"])))
        page += 1
        print(f"  listed page {page}, total {len(names)} objects", flush=True)
        token = js.get("nextPageToken")
        if not token:
            return names


def fetch(name: str, size: int) -> tuple[int, str]:
    rel = name[len(PREFIX):]
    path = DEST.joinpath(*rel.split("/"))
    if path.exists() and path.stat().st_size == size:
        return size, "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://storage.googleapis.com/{BUCKET}/{quote(name)}"

    def download():
        tmp = path.with_suffix(path.suffix + ".part")
        with session.get(url, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if tmp.stat().st_size != size:
            raise IOError(f"size mismatch for {name}: {tmp.stat().st_size} != {size}")
        tmp.replace(path)

    try:
        _retry(download)
        return size, "ok"
    except Exception:
        return size, "fail"


def main() -> int:
    print(f"Listing gs://{BUCKET}/{PREFIX} ...", flush=True)
    blobs = _retry(list_objects)
    total = sum(s for _, s in blobs)
    print(f"{len(blobs)} objects, {total / 1e9:.2f} GB", flush=True)
    DEST.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    done_bytes = 0
    failed: list[tuple[str, int]] = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(THREADS) as ex:
        futs = {ex.submit(fetch, n, s): (n, s) for n, s in blobs}
        for i, fut in enumerate(as_completed(futs), 1):
            size, status = fut.result()
            counts[status] += 1
            if status == "fail":
                failed.append(futs[fut])
            else:
                done_bytes += size
            if i % 250 == 0 or i == len(blobs):
                rate = done_bytes / max(time.monotonic() - t0, 1) / 1e6
                print(
                    f"{i}/{len(blobs)} files, {done_bytes / 1e9:.2f}/"
                    f"{total / 1e9:.2f} GB, {rate:.1f} MB/s, fails={counts['fail']}",
                    flush=True,
                )

    # sequential retry pass over whatever failed
    for item in list(failed):
        size, status = fetch(*item)
        if status != "fail":
            failed.remove(item)
            counts["ok"] += 1
            done_bytes += size

    print(
        "DONE." if not failed else f"FAILED: {len(failed)} files remain",
        flush=True,
    )
    for name, _ in failed[:20]:
        print("  failed:", name, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
