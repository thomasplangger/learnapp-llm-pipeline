import os
from .heuristic import HeuristicProvider
try:
    from .openai_provider import OpenAIProvider
except Exception:
    OpenAIProvider = None

def get_provider():
    provider = (os.getenv("AI_PROVIDER") or "").lower().strip()
    if provider == "heuristic":
        return HeuristicProvider()
    if provider == "openai" and os.getenv("OPENAI_API_KEY") and OpenAIProvider:
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("OPENAI_API_KEY") and OpenAIProvider:
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    return HeuristicProvider()

AI = get_provider()
