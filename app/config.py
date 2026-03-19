from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from pydantic import Field, field_validator, model_validator
from urllib.parse import quote


class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Multi-Tenant Agentic RAG Platform"
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = Field(default="replace-with-a-unique-32-character-minimum-secret", min_length=32)
    ALGORITHM: str = "HS256"
    CORS_ORIGINS: str = "http://localhost:8501"
    AUTH_MODE: Literal["local", "oidc"] = "local"
    OIDC_JWKS_URL: str = ""
    OIDC_ISSUER: str = ""
    OIDC_AUDIENCE: str = ""
    OIDC_TENANT_CLAIM: str = "tenant_id"
    OIDC_ROLE_CLAIM: str = "role"
    OIDC_ALGORITHMS: str = "RS256"
    PII_PROVIDER: Literal["regex", "presidio"] = "regex"
    REDACT_DOCUMENT_PII: bool = False
    PROMPT_GUARDRAILS_PROVIDER: Literal["regex", "nemo"] = "regex"
    NEMO_CHECK_MODE: Literal["all", "suspicious"] = "all"
    NEMO_GUARDRAILS_CONFIG_PATH: str = "guardrails"
    BOOTSTRAP_ADMIN_USERNAME: str = ""
    BOOTSTRAP_ADMIN_PASSWORD: str = Field(default="", max_length=72)
    BOOTSTRAP_ADMIN_TENANT_ID: str = ""
    ALLOW_SELF_REGISTRATION: bool = False
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, ge=5, le=1440)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=90)
    MAX_FAILED_LOGIN_ATTEMPTS: int = Field(default=5, ge=1, le=20)
    ACCOUNT_LOCK_MINUTES: int = Field(default=15, ge=1, le=1440)
    MAX_UPLOAD_BYTES: int = Field(default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    AUTH_RATE_LIMIT_PER_MINUTE: int = Field(default=10, ge=1, le=1000)
    CHAT_RATE_LIMIT_PER_MINUTE: int = Field(default=30, ge=1, le=1000)
    MAX_DOCUMENTS_PER_TENANT: int = Field(default=1000, ge=1, le=1000000)
    MAX_STORAGE_BYTES_PER_TENANT: int = Field(default=10 * 1024 * 1024 * 1024, ge=1024)
    REQUIRE_REDIS: bool = False

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    STRICT_RAG_MODE: bool = True
    USE_LLM_GRADER: bool = Field(default=False, description="Enable LLM-as-a-Judge relevance grading in LangGraph (set False for sub-second fast heuristic grading)")
    ENABLE_QUERY_REWRITE: bool = Field(default=False, description="Enable iterative query rewriting in LangGraph (set False for ultra-fast sub-500ms single-pass RAG)")

    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-v3.5"
    USE_COHERE_RERANK: bool = Field(default=False, description="Enable Cohere cross-encoder reranking on retrieved chunks")
    COHERE_TOP_N: int = Field(default=5, ge=1, le=50, description="Number of top reranked chunks to retain")

    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "rag_password"
    POSTGRES_DB: str = "rag_db"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_URL: str = ""

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    @field_validator("ALGORITHM")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        """Restrict JWT signing to the supported asymmetric-free algorithm."""
        normalized_value = value.strip().upper()
        if normalized_value != "HS256":
            raise ValueError("ALGORITHM must be HS256.")
        return normalized_value

    @model_validator(mode="after")
    def validate_deployment_configuration(self) -> "Settings":
        """Reject incomplete identity configuration before application startup."""
        if self.AUTH_MODE == "oidc":
            if not self.OIDC_JWKS_URL or not self.OIDC_ISSUER or not self.OIDC_AUDIENCE:
                raise ValueError(
                    "OIDC_JWKS_URL, OIDC_ISSUER, and OIDC_AUDIENCE are required in OIDC mode."
                )
        if self.ENVIRONMENT.lower() == "production":
            if self.ALLOW_SELF_REGISTRATION:
                raise ValueError("ALLOW_SELF_REGISTRATION must be false in production.")
            if self.SECRET_KEY.startswith("replace-with-"):
                raise ValueError("SECRET_KEY must be replaced with a secure secret in production.")
        return self

    @property
    def cors_origins(self) -> list[str]:
        """Return the explicitly configured browser origins for CORS."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def langfuse_configured(self) -> bool:
        """Return whether Langfuse telemetry export is configured."""
        return bool(self.LANGFUSE_PUBLIC_KEY.strip() and self.LANGFUSE_SECRET_KEY.strip())

    @property
    def bootstrap_admin_configured(self) -> bool:
        """Return whether all required bootstrap administrator values are configured."""
        values = (
            self.BOOTSTRAP_ADMIN_USERNAME,
            self.BOOTSTRAP_ADMIN_PASSWORD,
            self.BOOTSTRAP_ADMIN_TENANT_ID,
        )
        return all(value.strip() for value in values)

    @property
    def oidc_algorithms(self) -> list[str]:
        """Return the explicitly allowed OIDC signing algorithms."""
        algorithms = [item.strip().upper() for item in self.OIDC_ALGORITHMS.split(",") if item.strip()]
        if not algorithms or any(item not in {"RS256", "ES256"} for item in algorithms):
            raise ValueError("OIDC_ALGORITHMS may only contain RS256 or ES256.")
        return algorithms

    @property
    def postgres_dsn(self) -> str:
        user = quote(self.POSTGRES_USER, safe="")
        password = quote(self.POSTGRES_PASSWORD, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            f"?connect_timeout=10"
        )

    @property
    def async_postgres_dsn(self) -> str:
        user = quote(self.POSTGRES_USER, safe="")
        password = quote(self.POSTGRES_PASSWORD, safe="")
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
