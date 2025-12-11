#!/usr/bin/env python3
"""
Download missing paper PDFs from arXiv.

This script downloads all paper PDFs referenced in config.py that are missing.
"""
import os
import sys
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PAPER_REFERENCES

REFERENCES_DIR = Path(__file__).parent.parent / "references"
REFERENCES_DIR.mkdir(exist_ok=True)


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> bool:
    """
    Download PDF from arXiv.

    Args:
        arxiv_id: arXiv ID (e.g., "1706.06083")
        output_path: Path to save PDF

    Returns:
        True if successful, False otherwise
    """
    # Remove version suffix if present
    arxiv_id = arxiv_id.split('v')[0]

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        print(f"Downloading {arxiv_id} from {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {arxiv_id}: {e}")
        return False


def main():
    """Download all missing papers."""
    print("Checking for missing papers...")
    print(f"References directory: {REFERENCES_DIR}\n")

    missing = []
    existing = []

    for key, ref in PAPER_REFERENCES.items():
        arxiv_id = ref.get("arxiv", "")
        if not arxiv_id:
            continue

        # Remove version suffix
        arxiv_id_clean = arxiv_id.split('v')[0]
        pdf_path = REFERENCES_DIR / f"{arxiv_id_clean}.pdf"

        if pdf_path.exists():
            existing.append(arxiv_id_clean)
            print(f"✓ {arxiv_id_clean} already exists")
        else:
            missing.append((arxiv_id_clean, pdf_path, ref))

    print(f"\nFound {len(existing)} existing papers, {len(missing)} missing\n")

    if not missing:
        print("All papers are already downloaded!")
        return

    print("Downloading missing papers...\n")

    for arxiv_id, pdf_path, ref in missing:
        print(f"Paper: {ref['title']}")
        print(f"Authors: {ref['authors']} ({ref['year']})")
        success = download_arxiv_pdf(arxiv_id, pdf_path)
        print()

        if not success:
            print(f"⚠ Could not download {arxiv_id}")
            print(f"  Try manually: {ref.get('url', 'N/A')}\n")


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: requests library not installed.")
        print("Install with: pip install requests")
        sys.exit(1)

    main()

