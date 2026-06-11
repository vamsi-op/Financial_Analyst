"""
Push changed files to HF Space via raw HTTP (avoids httpx WinError 10054).
Usage: python push_files_hf.py
"""
import base64, json, os, sys, time
import requests

HF_TOKEN  = os.environ.get("HF_TOKEN", "")  # set via env or pass --token arg
REPO_ID   = "vamsi-op/Financial-Analyst"
HEADERS   = {"Authorization": f"Bearer {HF_TOKEN}"}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# (local_path, repo_path)
FILES_TO_PUSH = [
    ("app/frontend/index.html",     "app/frontend/index.html"),
    ("app/api/main.py",             "app/api/main.py"),
    ("app/api/routes.py",           "app/api/routes.py"),
    ("Dockerfile",                  "Dockerfile"),
    ("requirements-hf.txt",         "requirements.txt"),   # HF uses requirements.txt
]


def upload_file(local_path: str, repo_path: str) -> bool:
    full_path = os.path.join(PROJECT_ROOT, local_path)
    with open(full_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode()

    r = requests.post(
        f"https://huggingface.co/api/spaces/{REPO_ID}/commit/main",
        headers={**HEADERS, "Content-Type": "application/json"},
        data=json.dumps({
            "summary": f"feat: HTML/JS frontend, remove Streamlit ({repo_path})",
            "files": [{"path": repo_path, "encoding": "base64", "content": encoded}]
        }),
        timeout=90,
    )
    print(f"  {repo_path}: HTTP {r.status_code}")
    if r.status_code not in (200, 201):
        print(f"    Response: {r.text[:300]}")
    return r.status_code in (200, 201)


if __name__ == "__main__":
    print("Pushing files to HF Space…")
    ok = 0
    for local, remote in FILES_TO_PUSH:
        print(f"  Uploading {local} → {remote}")
        if upload_file(local, remote):
            ok += 1
        time.sleep(1.5)

    print(f"\n{ok}/{len(FILES_TO_PUSH)} files pushed.")
    if ok == len(FILES_TO_PUSH):
        print("Space will rebuild in ~2-3 min (new Docker image).")
        print("URL: https://huggingface.co/spaces/vamsi-op/Financial-Analyst")
        print("App: https://vamsi-op-financial-analyst.hf.space")
