"""
Push project to HuggingFace Spaces.

Usage: python scripts/push_to_hf.py

This script:
1. Pushes the repo to HF Spaces via the huggingface_hub API
2. Re-uploads the HF-specific README (with YAML frontmatter)
   because force push overwrites it

The GitHub README must NOT have YAML frontmatter (it shows as an ugly table).
The HF README is a separate file (hf_readme.md) with the frontmatter HF needs.
"""

import os
from huggingface_hub import HfApi, upload_folder

REPO_ID = "architechs/insurance-reshopping-predictor"
REPO_TYPE = "space"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def push():
    api = HfApi()

    print(f"Pushing to https://huggingface.co/spaces/{REPO_ID}")

    # Upload the entire project (including models and data for the app to work)
    upload_folder(
        folder_path=PROJECT_ROOT,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        ignore_patterns=[
            ".git/*",
            ".github/*",
            "docs/*",
            "notebooks/*",
            "hf_readme.md",
            "scripts/push_to_hf.py",
            "__pycache__/*",
            "*.pyc",
            "data/raw/test.csv",
            "data/raw/sample_submission.csv",
        ],
    )

    # Re-upload the HF-specific README (force push overwrites it)
    print("Re-uploading HF README with YAML frontmatter...")
    api.upload_file(
        path_or_fileobj=os.path.join(PROJECT_ROOT, "hf_readme.md"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print(f"Done! View at: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    push()
