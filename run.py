"""
Entry point for the Multi-Agent Financial Analyst.

Usage:
    python run.py api          # Start the FastAPI backend
    python run.py frontend     # Start the Streamlit dashboard
    python run.py both         # Start both (API in background)
    python run.py config       # Generate default config.json
    python run.py check        # Check system prerequisites
"""

import sys
import subprocess
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

    # 2. Ollama
    try:
        import ollama as _ollama
        models = _ollama.list()
        model_names = [m.model for m in models.models]
        ok = len(model_names) > 0
        detail = ", ".join(model_names[:5]) if ok else "No models pulled"
        checks.append(("Ollama + Models", ok, detail))
    except Exception as e:
        checks.append(("Ollama + Models", False, f"Not running: {e}"))

    # 3. PaddleOCR
    try:
        from paddleocr import PaddleOCR as _PaddleOCR
        checks.append(("PaddleOCR", True, "Installed"))
    except ImportError:
        checks.append(("PaddleOCR", False, "pip install paddleocr paddlepaddle"))

    # 4. FAISS
    try:
        import faiss as _faiss
        checks.append(("FAISS", True, f"Installed (version available)"))
    except ImportError:
        checks.append(("FAISS", False, "pip install faiss-cpu"))

    # 5. Key packages
    for pkg_name, pip_name in [
        ("pdfplumber", "pdfplumber"),
        ("fitz", "PyMuPDF"),
        ("streamlit", "streamlit"),
        ("fastapi", "fastapi"),
        ("langgraph", "langgraph"),
        ("sentence_transformers", "sentence-transformers"),
        ("plotly", "plotly"),
        ("pandas", "pandas"),
    ]:
        try:
            __import__(pkg_name)
            checks.append((pip_name, True, "Installed"))
        except ImportError:
            checks.append((pip_name, False, f"pip install {pip_name}"))

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
        print("  [OK] All prerequisites met! Ready to run.")
    else:
        print("  [!!] Some prerequisites are missing. Install them and re-check.")
    print("=" * 60)

    return all_ok


def start_api():
    """Start the FastAPI backend server."""
    from app.config import get_config
    config = get_config()

    print(f"Starting FastAPI server on http://{config.api.host}:{config.api.port}")
    print("API docs: http://localhost:8000/docs")

    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
    )


def start_frontend():
    """Start the Streamlit dashboard."""
    import os
    frontend_path = os.path.join(
        os.path.dirname(__file__), "app", "frontend", "app.py"
    )
    print(f"Starting Streamlit dashboard...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", frontend_path,
         "--server.port", "8501",
         "--server.headless", "true"],
    )


def start_both():
    """Start both API and frontend."""
    import threading

    # Start API in a background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    import time
    time.sleep(2)  # Give API time to start

    # Start frontend in the main thread
    start_frontend()


def generate_config():
    """Generate a default config.json file."""
    from app.config import save_default_config
    save_default_config("config.json")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "api":
        start_api()
    elif command == "frontend":
        start_frontend()
    elif command == "both":
        start_both()
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
