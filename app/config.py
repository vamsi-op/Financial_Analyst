"""
Central configuration for the Financial Analyst application.

All settings are configurable via environment variables or config.yaml.
No code changes needed to switch models, paths, or thresholds.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================
# Directory Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "sample"
VECTOR_DIR = DATA_DIR / "vectors"
REPORT_DIR = DATA_DIR / "reports"

# Ensure directories exist
for _dir in [DATA_DIR, UPLOAD_DIR, SAMPLE_DIR, VECTOR_DIR, REPORT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# LLM Configuration
# ============================================================
@dataclass
class LLMConfig:
    """Configuration for the Ollama LLM backend."""

    # Primary model
    primary_model: str = "qwen3:8b"

    # Fallback models (tried in order if primary fails)
    fallback_models: list = field(
        default_factory=lambda: ["llama3.1:8b", "mistral:7b"]
    )

    # Ollama server
    ollama_base_url: str = "http://localhost:11434"

    # Generation parameters
    temperature: float = 0.1  # Low temp for deterministic financial analysis
    num_ctx: int = 8192       # Context window size
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    # Timeouts
    request_timeout: int = 120  # seconds

    @property
    def all_models(self) -> list:
        """Return primary + fallback models in priority order."""
        return [self.primary_model] + self.fallback_models


# ============================================================
# OCR Configuration
# ============================================================
@dataclass
class OCRConfig:
    """Configuration for PaddleOCR."""

    engine: str = "paddleocr"
    language: str = "en"
    use_gpu: bool = False       # Default CPU; auto-detect available
    gpu_mem: int = 2048         # MB - conservative for 4GB VRAM
    use_angle_cls: bool = True  # Detect text orientation
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6      # Batch size for recognition
    show_log: bool = False      # Suppress PaddleOCR logs

    # Threshold for deciding if a PDF page needs OCR
    # If extracted text is shorter than this, apply OCR
    min_text_length: int = 50


# ============================================================
# Embedding & Vector Store Configuration
# ============================================================
@dataclass
class EmbeddingConfig:
    """Configuration for sentence embeddings and FAISS."""

    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # Must match model output dimension
    use_gpu: bool = False
    batch_size: int = 32

    # Chunking
    chunk_size: int = 512      # characters per chunk
    chunk_overlap: int = 50    # character overlap between chunks

    # Retrieval
    top_k: int = 5             # Number of chunks to retrieve


# ============================================================
# Risk Analysis Configuration
# ============================================================
@dataclass
class RiskConfig:
    """Weights and thresholds for the risk scoring algorithm."""

    # Risk category weights (must sum to 1.0)
    weight_revenue_decline: float = 0.25
    weight_profit_decline: float = 0.20
    weight_margin_compression: float = 0.15
    weight_debt_increase: float = 0.15
    weight_cash_flow: float = 0.15
    weight_sentiment: float = 0.10

    # Severity thresholds (percentage changes that trigger scoring)
    revenue_decline_threshold: float = -5.0    # % decline triggers risk
    profit_decline_threshold: float = -10.0
    margin_compression_threshold: float = -2.0  # percentage points
    debt_increase_threshold: float = 10.0       # % increase
    negative_cash_flow_threshold: float = 0.0   # below zero

    # Risk level boundaries
    low_risk_max: int = 30
    medium_risk_max: int = 60
    high_risk_max: int = 80
    # Above high_risk_max = critical


# ============================================================
# API Configuration
# ============================================================
@dataclass
class APIConfig:
    """Configuration for the FastAPI backend."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 50



# ============================================================
# Master Configuration
# ============================================================
@dataclass
class AppConfig:
    """Master application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration.

    Priority: environment variables > config file > defaults.

    Args:
        config_path: Optional path to a JSON config file.

    Returns:
        AppConfig instance with merged settings.
    """
    config = AppConfig()

    # --- Load from config file if provided ---
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
            _apply_dict_to_config(config, file_config)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")

    # --- Override from environment variables ---
    env_mappings = {
        "FA_MODEL": ("llm", "primary_model"),
        "FA_OLLAMA_URL": ("llm", "ollama_base_url"),
        "FA_TEMPERATURE": ("llm", "temperature"),
        "FA_NUM_CTX": ("llm", "num_ctx"),
        "FA_OCR_ENGINE": ("ocr", "engine"),
        "FA_OCR_GPU": ("ocr", "use_gpu"),
        "FA_EMBEDDING_MODEL": ("embedding", "model_name"),
        "FA_EMBEDDING_GPU": ("embedding", "use_gpu"),
        "FA_API_HOST": ("api", "host"),
        "FA_API_PORT": ("api", "port"),
        "FA_LOG_LEVEL": (None, "log_level"),
    }

    for env_var, (section, attr) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            target = getattr(config, section) if section else config
            current = getattr(target, attr)
            # Cast to the same type as the default
            if isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(target, attr, value)

    # --- Configure logging ---
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format=config.log_format,
    )

    return config


def _apply_dict_to_config(config: AppConfig, data: dict) -> None:
    """Recursively apply dictionary values to config dataclass fields."""
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                # Nested dataclass
                for sub_key, sub_value in value.items():
                    if hasattr(attr, sub_key):
                        setattr(attr, sub_key, sub_value)
            else:
                setattr(config, key, value)


def save_default_config(path: str = "config.json") -> None:
    """Save the default configuration to a JSON file for reference."""
    import dataclasses

    config = AppConfig()

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    with open(path, "w") as f:
        json.dump(_to_dict(config), f, indent=2)
    print(f"Default config saved to {path}")


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        config_path = os.environ.get("FA_CONFIG_PATH", "config.json")
        _config = load_config(config_path)
    return _config
