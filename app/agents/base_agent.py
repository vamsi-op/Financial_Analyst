"""
Abstract base class for all financial analysis agents.

Provides shared infrastructure for:
- LLM invocation with automatic model fallback
- JSON response parsing (handles markdown code fences)
- Consistent logging and error handling
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import AppConfig, get_config
from app.graph.state import PipelineState
from app.utils.llm import get_llm, invoke_with_fallback


class BaseAgent(ABC):
    """
    Abstract base class that every agent in the pipeline must extend.

    Responsibilities:
    - Manages a per-agent logger scoped under ``agent.<name>``.
    - Caches the global ``AppConfig`` so subclasses don't re-fetch it.
    - Offers ``_call_llm`` for LLM interaction with fallback.
    - Offers ``_parse_json_response`` for safe JSON extraction from LLM output.

    Subclasses **must** implement :meth:`invoke`.
    """

    def __init__(self, name: str, config: Optional[AppConfig] = None) -> None:
        """
        Initialise the agent.

        Args:
            name: Human-readable agent name (e.g. ``"kpi_agent"``).
            config: Application config.  Falls back to the global singleton.
        """
        self.name = name
        self.config: AppConfig = config or get_config()
        self.logger: logging.Logger = logging.getLogger(f"agent.{name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @abstractmethod
    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Process the pipeline state and return the updated state.

        Each agent reads the fields it needs, performs its analysis, writes
        its outputs back onto the state, and returns it.

        Args:
            state: The current shared pipeline state.

        Returns:
            The updated ``PipelineState``.
        """
        ...

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """
        Call the LLM with automatic fallback across configured models.

        Constructs a ``[SystemMessage, HumanMessage]`` pair and delegates
        to :func:`app.utils.llm.invoke_with_fallback`.

        Args:
            prompt: The user/human prompt text.
            system_prompt: Optional system-level instruction.

        Returns:
            The LLM's response content as a plain string.

        Raises:
            RuntimeError: If all models fail (propagated from
                ``invoke_with_fallback``).
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        self.logger.debug(
            "Calling LLM – prompt length=%d, system_prompt length=%d",
            len(prompt),
            len(system_prompt),
        )

        try:
            response = invoke_with_fallback(
                messages=messages,
                config=self.config.llm,
            )
            self.logger.debug(
                "LLM response received – length=%d", len(response)
            )
            return response
        except RuntimeError:
            self.logger.error(
                "All LLM models failed for agent '%s'", self.name
            )
            raise

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_json_response(self, response: str) -> dict:
        """
        Extract and parse JSON from an LLM response.

        Handles common LLM quirks:
        - Markdown code fences.
        - Leading/trailing whitespace and stray text outside the JSON.
        - Unescaped newlines / quotes inside string values.
        - Trailing commas before } or ].
        - Truncated JSON (tries to recover partial data).

        Returns an empty dict (and logs a warning) on complete failure.
        """
        if not response:
            self.logger.warning("Empty LLM response – returning empty dict")
            return {}

        text = response.strip()

        # 1. Strip markdown code fence
        fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
        match = fence_pattern.search(text)
        if match:
            text = match.group(1).strip()

        # 2. Find outermost { … }
        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                text = text[brace_start: brace_end + 1]

        # 3. Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 4. Repair common issues and retry
        repaired = self._repair_json(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            self.logger.warning(
                "Failed to parse JSON from LLM response: %s – raw snippet: %.300s",
                exc, response,
            )

        # 5. Last resort: extract individual string fields with regex
        return self._extract_fields_with_regex(response)

    @staticmethod
    def _repair_json(text: str) -> str:
        """Apply heuristic repairs to malformed JSON strings."""
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Replace literal newlines inside quoted strings with \n
        # (walk char by char to avoid mangling structure)
        result = []
        in_string = False
        escaped = False
        for ch in text:
            if escaped:
                result.append(ch)
                escaped = False
            elif ch == '\\':
                result.append(ch)
                escaped = True
            elif ch == '"':
                result.append(ch)
                in_string = not in_string
            elif in_string and ch == '\n':
                result.append('\\n')
            elif in_string and ch == '\r':
                pass  # strip CR
            else:
                result.append(ch)
        text = ''.join(result)

        # If JSON is truncated, try to close open brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        # Close any open string first
        if in_string:
            text += '"'
        text += ']' * max(0, open_brackets)
        text += '}' * max(0, open_braces)

        return text

    @staticmethod
    def _extract_fields_with_regex(text: str) -> dict:
        """
        Last-resort field extraction using regex.
        Pulls out key:"value" pairs for known summary fields.
        """
        result = {}
        # Match "key": "value" patterns
        pattern = re.compile(r'"(\w+)"\s*:\s*"((?:[^"]|\\.)*)"', re.DOTALL)
        for m in pattern.finditer(text):
            result[m.group(1)] = m.group(2).replace('\\n', '\n').replace('\\"', '"')
        # Match "key": ["item1", "item2"] patterns
        list_pattern = re.compile(r'"(\w+)"\s*:\s*\[([^\]]*?)\]', re.DOTALL)
        for m in list_pattern.finditer(text):
            items = re.findall(r'"((?:[^"]|\\.)*)"', m.group(2))
            if items:
                result[m.group(1)] = items
        return result

