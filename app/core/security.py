import re
import logging
from functools import lru_cache
from typing import Any, Protocol, cast

from app.config import settings

logger = logging.getLogger(__name__)


class _Analyzer(Protocol):
    """Type contract for the optional Presidio analyzer."""

    def analyze(self, *args: Any, **kwargs: Any) -> Any:
        """Analyze text for PII entities."""
        raise NotImplementedError


class _Anonymizer(Protocol):
    """Type contract for the optional Presidio anonymizer."""

    def anonymize(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Anonymize analyzed PII entities."""
        raise NotImplementedError


class _Rails(Protocol):
    """Type contract for the optional NeMo Guardrails runtime."""

    async def check_async(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run configured rails asynchronously."""
        raise NotImplementedError

# Comprehensive OWASP LLM Prompt Injection & Jailbreak Signatures
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt\s+(override|leak|reveal|dump)",
    r"you\s+are\s+now\s+(DAN|in\s+developer\s+mode|unfiltered|jailbroken)",
    r"bypass\s+(safety|security|content)\s+filters",
    r"jailbreak",
    r"disregard\s+(all\s+)?guidelines",
    r"repeat\s+the\s+words\s+above",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
]

# Sensitive PII & Secret token redaction regexes
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
SSN_REGEX = r"\b\d{3}-\d{2}-\d{4}\b"
CREDIT_CARD_REGEX = r"\b(?:\d{4}[ -]?){3}\d{4}\b"
JWT_REGEX = r"ey[A-Za-z0-9_-]{10,}\.ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-+/=]{10,}"
API_KEY_REGEX = r"\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36})\b"


class SecurityGuardrails:
    @staticmethod
    @lru_cache(maxsize=1)
    def _presidio_engines() -> tuple[_Analyzer, _Anonymizer]:
        """Load Presidio engines only when the configured provider requires them."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as error:
            raise RuntimeError(
                "PII_PROVIDER=presidio requires presidio-analyzer and presidio-anonymizer."
            ) from error
        return cast(_Analyzer, AnalyzerEngine()), cast(_Anonymizer, AnonymizerEngine())

    @staticmethod
    def _redact_with_presidio(text: str) -> str:
        """Redact recognized PII using Presidio's analyzer and anonymizer."""
        analyzer, anonymizer = SecurityGuardrails._presidio_engines()
        results = analyzer.analyze(text=text, language="en")
        result = anonymizer.anonymize(text=text, analyzer_results=results)
        return result.text

    @staticmethod
    @lru_cache(maxsize=1)
    def _nemo_rails() -> _Rails:
        """Load the configured NeMo Guardrails input-rail runtime once."""
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as error:
            raise RuntimeError(
                "PROMPT_GUARDRAILS_PROVIDER=nemo requires nemoguardrails."
            ) from error
        config = RailsConfig.from_path(settings.NEMO_GUARDRAILS_CONFIG_PATH)
        return cast(_Rails, LLMRails(config))

    @staticmethod
    def inspect_prompt(prompt: str) -> None:
        """Inspects input prompt for malicious prompt injection patterns."""
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"🚨 Potential Prompt Injection Intercepted: {pattern}")
                raise ValueError("Security Policy Violation: Malicious prompt pattern detected.")

    @staticmethod
    def redact_pii(text: str) -> str:
        """Redacts sensitive PII and credential patterns from text and response streams."""
        if settings.PII_PROVIDER == "presidio":
            return SecurityGuardrails._redact_with_presidio(text)
        redacted = re.sub(EMAIL_REGEX, "[REDACTED_EMAIL]", text)
        redacted = re.sub(PHONE_REGEX, "[REDACTED_PHONE]", redacted)
        redacted = re.sub(SSN_REGEX, "[REDACTED_SSN]", redacted)
        redacted = re.sub(CREDIT_CARD_REGEX, "[REDACTED_CARD]", redacted)
        redacted = re.sub(JWT_REGEX, "[REDACTED_JWT]", redacted)
        redacted = re.sub(API_KEY_REGEX, "[REDACTED_SECRET]", redacted)
        return redacted

    @classmethod
    def sanitize_prompt(cls, prompt: str) -> str:
        """Block malicious prompts and redact PII before any downstream processing."""
        cls.inspect_prompt(prompt)
        return cls.redact_pii(prompt)

    @classmethod
    async def sanitize_prompt_async(cls, prompt: str) -> str:
        """Run synchronous checks and optional NeMo input rails before model access."""
        sanitized = cls.sanitize_prompt(prompt)
        if settings.PROMPT_GUARDRAILS_PROVIDER != "nemo":
            return sanitized
        try:
            from nemoguardrails.rails.llm.options import RailType
        except ImportError as error:
            raise RuntimeError(
                "PROMPT_GUARDRAILS_PROVIDER=nemo requires nemoguardrails."
            ) from error
        result = await cls._nemo_rails().check_async(
            messages=[{"role": "user", "content": sanitized}],
            rail_types=[RailType.INPUT],
        )
        status_value = str(getattr(result, "status", "")).upper()
        if "BLOCK" in status_value:
            raise ValueError("Security Policy Violation: NeMo input rail blocked the prompt.")
        content = getattr(result, "content", None)
        return str(content) if content else sanitized

    @classmethod
    async def sanitize_output_async(cls, output_text: str) -> str:
        """Sanitize LLM output text with PII redaction and optional NeMo output rails."""
        sanitized = cls.redact_pii(output_text)
        if settings.PROMPT_GUARDRAILS_PROVIDER != "nemo":
            return sanitized
        try:
            from nemoguardrails.rails.llm.options import RailType
            result = await cls._nemo_rails().check_async(
                messages=[{"role": "assistant", "content": sanitized}],
                rail_types=[RailType.OUTPUT],
            )
            status_value = str(getattr(result, "status", "")).upper()
            if "BLOCK" in status_value:
                return "I cannot provide this response because it violates enterprise compliance policies."
            content = getattr(result, "content", None)
            return str(content) if content else sanitized
        except Exception as exc:
            logger.warning("NeMo output guardrail warning: %s", exc)
            return sanitized
