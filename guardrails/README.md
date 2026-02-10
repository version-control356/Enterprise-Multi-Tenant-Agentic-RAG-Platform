# NeMo Guardrails configuration

NeMo Guardrails is enabled by default in the project configuration. It runs the
`self check input` rail through the configured model and returns a blocked
status before retrieval or generation.

The guardrail model is configured independently from the primary Groq
`openai/gpt-oss-20b` generation model. Configure the credentials required by
`guardrails/config.yml` before sending prompts.

To use the lightweight fallback during offline development, set
`PROMPT_GUARDRAILS_PROVIDER=regex` and `PII_PROVIDER=regex`.
