"""Which models the app offers.

With an API key the lists come from the provider's models endpoint, so new
models show up without a code change. The FALLBACK lists are used without a
key or when the request fails, and their first entry is the default choice.
"""
import re

FALLBACK_CHAT_MODELS = {
    "claude": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
}
FALLBACK_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

# OpenAI lists every kind of model in one endpoint with no type field, so chat
# models have to be recognised by name.
_OPENAI_CHAT_PREFIX = re.compile(r"^(gpt-|o\d)")
_OPENAI_NOT_CHAT = re.compile(
    r"image|audio|realtime|transcribe|tts|live|search|codex|moderation|embedding|translate|instruct"
    r"|-pro\b"                # pro models are not served by the chat completions API
    r"|-\d{4}-\d{2}-\d{2}$"   # dated snapshots; the undated alias is listed too
)


def is_openai_chat_model(model_id: str) -> bool:
    return bool(_OPENAI_CHAT_PREFIX.match(model_id)) and not _OPENAI_NOT_CHAT.search(model_id)


def _openai_model_ids(api_key: str) -> list[str]:
    """All model ids of the account, newest first."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=10, max_retries=0)
    return [m.id for m in sorted(client.models.list(), key=lambda m: m.created, reverse=True)]


def fetch_chat_models(provider: str, api_key: str) -> list[str]:
    """Chat model ids the key can use, newest first. Raises when the request fails."""
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, timeout=10, max_retries=0)
        return [m.id for m in client.models.list(limit=100)]
    if provider == "openai":
        return [m for m in _openai_model_ids(api_key) if is_openai_chat_model(m)]
    raise ValueError(f"Unknown provider: {provider}")


def fetch_embedding_models(api_key: str) -> list[str]:
    """OpenAI embedding model ids the key can use. Raises when the request fails."""
    return [m for m in _openai_model_ids(api_key) if m.startswith("text-embedding-")]


def default_model(options: list[str], fallback: list[str]) -> str:
    """The first fallback model that is on offer, else the first option."""
    return next((m for m in fallback if m in options), options[0])
