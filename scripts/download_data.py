"""Download FI-2010 dataset from the official DeepLOB GitHub repository.

The FI-2010 dataset contains limit order book data from 5 Finnish stocks
traded on Nasdaq Nordic over 10 business days. The data includes:
- 40 LOB features (10 levels × 4 features: ask_price, ask_vol, bid_price, bid_vol)
- 104 hand-crafted features (not used by DeepLOB)
- 5 label rows (horizons k=10, 20, 30, 50, 100)

📚 Study this on Desktop: Limit Order Books — what are bid/ask prices,
   depth levels, and how does a continuous double auction market work?

Source: github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
"""

import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path


def download_fi2010(data_dir: str = "data/raw") -> None:
    """Download and extract the FI-2010 dataset.

    Downloads data.zip from the official DeepLOB GitHub repo and extracts
    the .txt files into data/raw/.

    Parameters
    ----------
    data_dir : str
        Directory to save the raw data files.
    """
    # Resolve paths relative to project root (one level up from scripts/)
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "data.zip"

    # URL for the data.zip bundled in the official DeepLOB repo
    url = (
        "https://raw.githubusercontent.com/zcakhaa/"
        "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
        "master/data/data.zip"
    )

    # Check if data is already downloaded
    # The key training file for Setup 2 (z-score normalization)
    check_file = raw_dir / "Train_Dst_NoAuction_ZScore_CF_7.txt"
    if check_file.exists():
        print(f"Data already exists at {raw_dir}")
        print(f"  Found: {check_file.name}")
        return

    # Download with progress reporting
    print(f"Downloading FI-2010 dataset from official DeepLOB repo...")
    print(f"  URL: {url}")
    print(f"  Destination: {zip_path}")

    # Handle SSL certificates (macOS Python sometimes needs certifi)
    try:
        import certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_context = ssl.create_default_context()

    try:
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req, context=ssl_context)
        total_size = int(response.headers.get("Content-Length", 0))

        # Download with progress bar
        downloaded = 0
        chunk_size = 8192
        with open(zip_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Downloaded: {mb:.1f} MB ({pct:.0f}%)", end="")
        print()  # newline after progress

    except urllib.error.URLError as e:
        print(f"ERROR: Failed to download data: {e}")
        print("  Try manually downloading from:")
        print(f"  {url}")
        sys.exit(1)

    # Extract the zip file
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # List contents
        file_list = zf.namelist()
        print(f"  Archive contains {len(file_list)} files:")
        for name in sorted(file_list):
            if name.endswith(".txt"):
                print(f"    {name}")

        # Extract all files into raw_dir (flattening any subdirectories)
        for member in file_list:
            # Skip directories and non-txt files
            if member.endswith("/") or not member.endswith(".txt"):
                continue
            # Extract to raw_dir with just the filename (no subdirs)
            filename = os.path.basename(member)
            target_path = raw_dir / filename
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    # Clean up zip file to save space
    zip_path.unlink()
    print(f"Removed {zip_path.name} (extracted files saved)")

    # Verify extraction
    txt_files = sorted(raw_dir.glob("*.txt"))
    print(f"\nExtracted {len(txt_files)} files to {raw_dir}:")
    for f in txt_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    print("\nFI-2010 download complete!")


if __name__ == "__main__":
    download_fi2010()
