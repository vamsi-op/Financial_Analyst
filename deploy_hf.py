"""
Deploy the Financial Analyst to Hugging Face Spaces.
Usage: python deploy_hf.py --token YOUR_HF_TOKEN
"""

import argparse
import os
import shutil
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--token", required=True, help="Hugging Face write token")
    parser.add_argument("--username", default="vamsi-op", help="HF username")
    parser.add_argument("--space-name", default="Financial-Analyst", help="Space name")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)
    repo_id = f"{args.username}/{args.space_name}"

    # ── 1. Create the Space (idempotent) ──────────────────────────────────────
    print(f"\n[1/4] Creating Space: {repo_id} (Docker SDK)...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=args.token,
        )
        print(f"      Space ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"      Note: {e}")

    # ── 2. Prepare a staging directory ────────────────────────────────────────
    print("\n[2/4] Preparing files for upload ...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    staging = tempfile.mkdtemp(prefix="hf_deploy_")

    # Files/dirs to include
    include = [
        "app",
        "data/sample",
        "data/generate_sample_reports.py",
        "config.json",
        "run.py",
        "README.md",
        "Dockerfile",
    ]

    for item in include:
        src = os.path.join(project_root, item)
        dst = os.path.join(staging, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            print(f"      + {item}/")
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"      + {item}")

    # Use the lightweight HF requirements (replaces requirements.txt)
    hf_req_src = os.path.join(project_root, "requirements-hf.txt")
    req_dst    = os.path.join(staging, "requirements.txt")
    shutil.copy2(hf_req_src, req_dst)
    print(f"      + requirements.txt  (from requirements-hf.txt)")

    # Create a .gitignore so HF doesn't complain
    with open(os.path.join(staging, ".gitignore"), "w") as f:
        f.write("*.pyc\n__pycache__/\n.env\ndata/uploads/\ndata/vectors/\n")
    print(f"      + .gitignore")

    # Create data/uploads and data/vectors placeholder dirs (needed at runtime)
    for d in ["data/uploads", "data/vectors", "data/reports"]:
        os.makedirs(os.path.join(staging, d), exist_ok=True)
        # Add .gitkeep so empty dirs are tracked
        open(os.path.join(staging, d, ".gitkeep"), "w").close()
    print(f"      + data/uploads/, data/vectors/, data/reports/ (placeholders)")

    # ── 3. Upload the folder ──────────────────────────────────────────────────
    print(f"\n[3/4] Uploading to {repo_id} ...")
    api.upload_folder(
        folder_path=staging,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy: Multi-Agent Financial Analyst with Groq support",
        ignore_patterns=["*.pyc", "__pycache__/*", ".env"],
    )
    print("      Upload complete!")

    # ── 4. Cleanup ────────────────────────────────────────────────────────────
    shutil.rmtree(staging, ignore_errors=True)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"""
[4/4] DEPLOYMENT COMPLETE!

  Space URL : https://huggingface.co/spaces/{repo_id}

  IMPORTANT — Add these Secrets in your Space settings:
  (Settings > Variables and Secrets > New Secret)

    GROQ_API_KEY  = gsk_...your key...
    LLM_PROVIDER  = groq
    GROQ_MODEL    = llama-3.1-8b-instant

  The Space will auto-restart after you add the secrets.
""")


if __name__ == "__main__":
    main()
