"""
Entry point for the Multi-Agent Financial Analyst.

Usage:
    python run.py api       # Start the FastAPI + HTML frontend server
    python run.py config    # Generate default config.json
    python run.py check     # Check system prerequisites
"""

import sys
import logging

logger = logging.getLogger(__name__)


def check_prerequisites():
    """Verify all system prerequisites are met."""
    print("=" * 60)
    print("  Financial Analyst — System Check")
    print("=" * 60)

    checks = []

    # 1. Python version
    py_ver = sys.version_info
    ok = py_ver >= (3, 10)
    checks.append(("Python >= 3.10", ok, f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"))

    # 2. FAISS
    try:
        import faiss as _faiss  # noqa: F401
        checks.append(("FAISS", True, "Installed"))
    except ImportError:
        checks.append(("FAISS", False, "pip install faiss-cpu"))

    # 3. Key packages
    for pkg_name, pip_name in [
        ("pdfplumber", "pdfplumber"),
        ("fitz", "PyMuPDF"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("langgraph", "langgraph"),
        ("sentence_transformers", "sentence-transformers"),
        ("groq", "groq"),
        ("pandas", "pandas"),
    ]:
        try:
            __import__(pkg_name)
            checks.append((pip_name, True, "Installed"))
        except ImportError:
            checks.append((pip_name, False, f"pip install {pip_name}"))

    # 4. Groq API key
    import os
    groq_key = os.environ.get("GROQ_API_KEY", "")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        groq_key = os.environ.get("GROQ_API_KEY", "")
    except ImportError:
        pass
    checks.append(("GROQ_API_KEY", bool(groq_key), "Set" if groq_key else "Missing — set in .env"))

    # Print results
    print()
    all_ok = True
    for name, ok, detail in checks:
        status = "[OK]" if ok else "[!!]"
        if not ok:
            all_ok = False
        print(f"  {status} {name:<25} {detail}")

    print()
    if all_ok:
        print("  [OK] All prerequisites met! Run: python run.py api")
    else:
        print("  [!!] Some prerequisites are missing. Install them and re-check.")
    print("=" * 60)

    return all_ok


def start_api():
    """Start the FastAPI backend (serves HTML frontend + API)."""
    from app.config import get_config
    config = get_config()

    print(f"Starting FastAPI server on http://{config.api.host}:{config.api.port}")
    print(f"Open your browser at: http://localhost:{config.api.port}")
    print("API docs: http://localhost:8000/docs\n")

    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
    )


def generate_config():
    """Generate a default config.json file."""
    from app.config import save_default_config
    save_default_config("config.json")
    print("Default config.json written.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "api":
        start_api()
    elif command == "config":
        generate_config()
    elif command == "check":
        check_prerequisites()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
