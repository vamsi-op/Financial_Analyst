"""
FAISS-backed vector store with SentenceTransformer embeddings.

Stores document chunks as dense vectors in a FAISS ``IndexFlatIP`` index
(inner-product on L2-normalised vectors ≡ cosine similarity).  Document
texts and optional metadata are kept in a sidecar JSON file because FAISS
itself stores only float vectors.

Usage::

    from app.config import get_config
    store = FAISSStore(get_config().embedding)
    store.add_documents(["chunk 1", "chunk 2"], [{"page": 1}, {"page": 2}])
    results = store.search("quarterly revenue growth", top_k=3)
    store.save("data/vectors/index")
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import EmbeddingConfig, get_config

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Vector store backed by FAISS and SentenceTransformers.

    Attributes:
        config:     Embedding configuration (model, dimension, batch_size).
        _model:     SentenceTransformer model instance.
        _index:     FAISS index (``IndexFlatIP`` for cosine similarity).
        _texts:     Parallel list of document texts.
        _metadata:  Parallel list of metadata dicts.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """
        Initialise the vector store.

        Loads the SentenceTransformer model and creates an empty FAISS index.

        Args:
            config: Embedding configuration.  Falls back to
                    ``get_config().embedding`` if not provided.

        Raises:
            ImportError: If ``sentence-transformers`` or ``faiss`` is missing.
        """
        self.config = config or get_config().embedding

        # --- Load embedding model ---
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s", self.config.model_name)
        device = "cuda" if self.config.use_gpu else "cpu"
        self._model = SentenceTransformer(
            self.config.model_name, device=device
        )
        logger.info(
            "Embedding model loaded (dim=%d, device=%s)",
            self.config.dimension,
            device,
        )

        # --- Initialise FAISS index ---
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu "
                "(or faiss-gpu for GPU support)"
            ) from exc

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.config.dimension)
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []

        logger.info("FAISSStore initialised (dimension=%d)", self.config.dimension)

    # ----------------------------------------------------------
    # Core operations
    # ----------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Encode texts and add them to the index.

        Args:
            texts:    List of text chunks to embed and store.
            metadata: Optional parallel list of metadata dicts.  If
                      shorter than *texts*, missing entries are filled
                      with empty dicts.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            logger.warning("add_documents called with empty text list")
            return

        # Pad metadata to match texts
        if metadata is None:
            metadata = [{} for _ in texts]
        elif len(metadata) < len(texts):
            metadata.extend({} for _ in range(len(texts) - len(metadata)))

        logger.info("Encoding %d documents (batch_size=%d)", len(texts), self.config.batch_size)

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2-normalise for cosine sim
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            logger.error("Embedding encoding failed: %s", exc)
            raise

        # Validate dimensions
        if embeddings.shape[1] != self.config.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: model produced {embeddings.shape[1]}, "
                f"but config.dimension is {self.config.dimension}"
            )

        self._index.add(embeddings)
        self._texts.extend(texts)
        self._metadata.extend(metadata)

        logger.info(
            "Added %d documents — total index size: %d",
            len(texts),
            self._index.ntotal,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Search the index for the closest matches to *query*.

        Args:
            query: The search query string.
            top_k: Number of results to return.  Defaults to
                   ``config.top_k``.

        Returns:
            A list of result dicts sorted by descending similarity::

                [
                    {"text": "...", "score": 0.87, "metadata": {...}},
                    ...
                ]
        """
        top_k = top_k or self.config.top_k

        if self._index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Clamp top_k
        top_k = min(top_k, self._index.ntotal)

        try:
            query_vec = self._model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vec = np.asarray(query_vec, dtype=np.float32)
        except Exception as exc:
            logger.error("Query encoding failed: %s", exc)
            return []

        scores, indices = self._index.search(query_vec, top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for missing results
            results.append({
                "text": self._texts[idx],
                "score": float(score),
                "metadata": self._metadata[idx],
            })

        logger.debug("Search returned %d results for query: '%s…'", len(results), query[:60])
        return results

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the index and metadata to disk.

        Creates two files:
        * ``<path>.index`` — the FAISS binary index.
        * ``<path>.json``  — texts and metadata.

        Args:
            path: Base path (without extension).
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        index_path = str(base.with_suffix(".index"))
        meta_path = str(base.with_suffix(".json"))

        try:
            self._faiss.write_index(self._index, index_path)
            logger.info("FAISS index saved to %s", index_path)
        except Exception as exc:
            logger.error("Failed to save FAISS index: %s", exc)
            raise

        try:
            sidecar = {
                "texts": self._texts,
                "metadata": self._metadata,
                "model_name": self.config.model_name,
                "dimension": self.config.dimension,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, ensure_ascii=False, indent=2)
            logger.info("Metadata saved to %s (%d entries)", meta_path, len(self._texts))
        except Exception as exc:
            logger.error("Failed to save metadata: %s", exc)
            raise

    def load(self, path: str) -> None:
        """
        Load a previously saved index and metadata from disk.

        Args:
            path: Base path (without extension) — same value passed to
                  :pymeth:`save`.

        Raises:
            FileNotFoundError: If the index or metadata file is missing.
            ValueError:        If the saved dimension doesn't match config.
        """
        base = Path(path)
        index_path = base.with_suffix(".index")
        meta_path = base.with_suffix(".json")

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        try:
            self._index = self._faiss.read_index(str(index_path))
            logger.info(
                "FAISS index loaded from %s (%d vectors)",
                index_path,
                self._index.ntotal,
            )
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            raise

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                sidecar = json.load(f)
            self._texts = sidecar.get("texts", [])
            self._metadata = sidecar.get("metadata", [])

            saved_dim = sidecar.get("dimension")
            if saved_dim and saved_dim != self.config.dimension:
                raise ValueError(
                    f"Dimension mismatch: saved index has dimension {saved_dim}, "
                    f"but current config specifies {self.config.dimension}"
                )

            logger.info(
                "Metadata loaded from %s (%d entries)", meta_path, len(self._texts)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to load metadata: %s", exc)
            raise

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def clear(self) -> None:
        """Reset the index, removing all stored vectors and metadata."""
        self._index = self._faiss.IndexFlatIP(self.config.dimension)
        self._texts.clear()
        self._metadata.clear()
        logger.info("FAISSStore cleared")

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"FAISSStore(model={self.config.model_name!r}, "
            f"dim={self.config.dimension}, "
            f"vectors={self._index.ntotal})"
        )
