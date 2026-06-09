"""
LLM provider abstraction — supports Groq, Ollama, and auto-detection.

Provider is resolved in this priority order:
  1. LLM_PROVIDER env var (or .env file)
  2. Groq  — if GROQ_API_KEY is set
  3. Ollama — if server is running
  4. Raises RuntimeError with helpful message

Usage is identical regardless of provider:
    from app.utils.llm import invoke_with_fallback
    response = invoke_with_fallback([("user", "Say hello")])
"""

import logging
import os
from typing import Optional

# Load .env file if present (picks up GROQ_API_KEY, LLM_PROVIDER, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # don't override vars already in environment
except ImportError:
    pass

from app.config import get_config, LLMConfig

logger = logging.getLogger(__name__)

# ── cached state ──────────────────────────────────────────────────────────────
_active_model: Optional[str] = None
_active_provider: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# Provider detection helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_groq_key() -> Optional[str]:
    return os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_KEY")


def _get_gemini_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _groq_available() -> bool:
    try:
        import groq as _groq_pkg  # noqa: F401
        return bool(_get_groq_key())
    except ImportError:
        return False


def check_ollama_running() -> bool:
    """Check if the Ollama server is running and reachable."""
    try:
        import ollama as _ollama
        _ollama.list()
        return True
    except Exception:
        return False


def _detect_provider() -> str:
    """
    Auto-detect the best available LLM provider.

    Priority: env var LLM_PROVIDER → Groq → Ollama
    """
    explicit = os.environ.get("LLM_PROVIDER", "").lower()
    if explicit in ("groq", "ollama", "gemini"):
        return explicit

    if _groq_available():
        logger.info("Auto-detected provider: groq (GROQ_API_KEY is set)")
        return "groq"

    if check_ollama_running():
        logger.info("Auto-detected provider: ollama")
        return "ollama"

    # Return groq/ollama placeholder — will raise at invoke time
    return "none"


# ═════════════════════════════════════════════════════════════════════════════
# Groq backend
# ═════════════════════════════════════════════════════════════════════════════

# Free-tier Groq models in preference order (current as of 2025)
_GROQ_MODELS = [
    "llama-3.1-8b-instant",      # fast, free, great for structured output
    "llama-3.3-70b-versatile",   # larger, more capable
    "llama3-groq-8b-8192-tool-use-preview",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


def _invoke_groq(messages: list, temperature: float = 0.1) -> str:
    """Call Groq API and return the response text."""
    import groq as groq_pkg

    api_key = _get_groq_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    client = groq_pkg.Groq(api_key=api_key)

    # Resolve model: env var → first available
    model = os.environ.get("GROQ_MODEL", _GROQ_MODELS[0])

    # Convert (role, content) tuples or LangChain messages to Groq format
    groq_msgs = []
    for m in messages:
        if isinstance(m, tuple):
            role, content = m
        elif hasattr(m, "type"):   # LangChain message object
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            role = role_map.get(m.type, "user")
            content = m.content
        else:
            role, content = "user", str(m)
        groq_msgs.append({"role": role, "content": content})

    # Try models in order
    errors = []
    candidates = [model] + [m for m in _GROQ_MODELS if m != model]
    for candidate in candidates:
        try:
            resp = client.chat.completions.create(
                model=candidate,
                messages=groq_msgs,
                temperature=temperature,
                max_tokens=4096,
            )
            global _active_model, _active_provider
            _active_model = candidate
            _active_provider = "groq"
            logger.info("Groq response received via model: %s", candidate)
            return resp.choices[0].message.content
        except Exception as e:
            err = f"{candidate}: {e}"
            logger.warning("Groq model %s failed: %s", candidate, e)
            errors.append(err)

    raise RuntimeError("All Groq models failed:\n" + "\n".join(errors))


# ═════════════════════════════════════════════════════════════════════════════
# Ollama backend (kept for backward compat)
# ═════════════════════════════════════════════════════════════════════════════

def list_available_models() -> list[str]:
    """List all models currently pulled in Ollama."""
    try:
        import ollama as _ollama
        response = _ollama.list()
        return [m.model for m in response.models]
    except Exception as e:
        logger.debug("Could not list Ollama models: %s", e)
        return []


def find_available_model(config: Optional[LLMConfig] = None) -> Optional[str]:
    """Find the first available Ollama model from the configured priority list."""
    if config is None:
        config = get_config().llm
    available = list_available_models()
    if not available:
        return None
    for model in config.all_models:
        if model in available:
            return model
        base = model.split(":")[0]
        for avail in available:
            if avail.startswith(base):
                return avail
    return available[0]


def _invoke_ollama(messages: list, temperature: float = 0.1) -> str:
    """Call Ollama via LangChain and return the response text."""
    from langchain_ollama import ChatOllama

    config = get_config().llm

    if not check_ollama_running():
        raise RuntimeError(
            "Ollama server is not running. Start it with 'ollama serve'."
        )

    model = find_available_model(config)
    if model is None:
        raise RuntimeError("No models available in Ollama. Run 'ollama pull qwen3:8b'.")

    llm = ChatOllama(
        model=model,
        base_url=config.ollama_base_url,
        temperature=temperature,
        num_ctx=config.num_ctx,
    )
    response = llm.invoke(messages)
    global _active_model, _active_provider
    _active_model = model
    _active_provider = "ollama"
    return response.content


# ═════════════════════════════════════════════════════════════════════════════
# Public interface
# ═════════════════════════════════════════════════════════════════════════════

def invoke_with_fallback(
    messages: list,
    config: Optional[LLMConfig] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Invoke the configured LLM provider with automatic fallback.

    Tries Groq first if available, then Ollama. Provider can be forced
    via the LLM_PROVIDER environment variable.

    Args:
        messages: List of (role, content) tuples or LangChain message objects.
        config:   LLM config override (used only for Ollama).
        temperature: Temperature override.

    Returns:
        The LLM response as a plain string.

    Raises:
        RuntimeError: If all configured providers fail.
    """
    temp = temperature if temperature is not None else 0.1
    provider = _detect_provider()

    logger.info("LLM invoke — provider=%s", provider)

    if provider == "groq":
        return _invoke_groq(messages, temperature=temp)

    if provider == "ollama":
        return _invoke_ollama(messages, temperature=temp)

    if provider == "gemini":
        raise RuntimeError(
            "Gemini provider not yet configured. "
            "Set GEMINI_API_KEY and install google-generativeai."
        )

    raise RuntimeError(
        "No LLM provider available.\n"
        "Options:\n"
        "  1. Set GROQ_API_KEY in your .env file (free at console.groq.com)\n"
        "  2. Install and start Ollama: https://ollama.ai\n"
        "  3. Set LLM_PROVIDER=groq in your .env file"
    )


def get_active_model() -> str:
    """Return the name of the currently active model."""
    if _active_model:
        return _active_model
    provider = _detect_provider()
    if provider == "groq":
        return os.environ.get("GROQ_MODEL", _GROQ_MODELS[0])
    if provider == "ollama":
        model = find_available_model()
        return model or "unknown"
    return "none"


def get_active_provider() -> str:
    """Return the name of the active provider (groq / ollama / none)."""
    return _active_provider or _detect_provider()


# Legacy: kept so existing code that does `get_llm()` doesn't break
def get_llm(model=None, temperature=None, num_ctx=None, force_new=False):
    """
    Backward-compatible shim. Returns a callable wrapper around
    invoke_with_fallback so agent code using `.invoke(messages)` still works.
    """
    class _LLMShim:
        def __init__(self, temp):
            self.temperature = temp or 0.1

        def invoke(self, messages):
            class _Resp:
                def __init__(self, content):
                    self.content = content
            return _Resp(invoke_with_fallback(messages, temperature=self.temperature))

    return _LLMShim(temperature)
