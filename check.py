import argparse
import io
import zipfile
from pathlib import Path

import requests

API = "https://ambientCG.com/api/v3/assets"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=500,
                    help="Max materials to download (ambientCG has ~2000)")
    ap.add_argument("--resolution", default="1K-JPG")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    offset, got = 0, 0
    page_size = 100
    while got < args.limit:
        r = requests.get(API, params={
            "type": "Material", "limit": page_size, "offset": offset,
            "include": "downloads",
        })
        r.raise_for_status()
        assets = r.json().get("assets", [])
        if not assets:
            break
        for a in assets:
            if got >= args.limit:
                break
            asset_id = a["id"]
            out_path = args.out / f"{asset_id}.jpg"
            if out_path.exists():
                got += 1
                continue
            zip_url = next(
                (d["url"] for d in a.get("downloads", [])
                 if d.get("attributes") == args.resolution
                 and d.get("extension") == "zip"),
                None,
            )
            if not zip_url:
                continue
            print(f"[{got+1}/{args.limit}] {asset_id}")
            blob = requests.get(zip_url).content
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for name in z.namelist():
                    if "_Color" in name and name.lower().endswith((".jpg", ".jpeg")):
                        out_path.write_bytes(z.read(name))
                        got += 1
                        break
        offset += page_size

if __name__ == "__main__":
    main()