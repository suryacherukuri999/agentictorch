"""Models the agent can use: free ones for development, OpenAI only for the final runs."""

from hotloop.agent.llm import Model

MODELS = {
    # Local and free, through Ollama on the Mac. Checked on ollama.com 2026-10-24:
    # tool calling, 6.6 GB download, fits the M4's 24 GB with room for context.
    # With greedy decoding it can reason until the token cap without answering;
    # a little temperature and low reasoning effort keep turns short.
    "ornith-9b": Model(
        "ornith-1.5:9b", "http://localhost:11434/v1", temperature=0.6, reasoning_effort="low"
    ),
    # Free tier through Google's OpenAI-compatible endpoint. Checked 2026-10-24:
    # gemini-2.5-flash is closed to new users, and free-tier calls took 9-80 s each.
    "gemini-flash": Model(
        "gemini-3.8-flash",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
        temperature=1.0,
    ),
    # Paid, for the final comparison only. Checked on developers.openai.com 2026-10-24:
    # $0.10 per million input tokens and $0.50 per million output tokens, and Chat
    # Completions allows tool calls only with reasoning effort "none".
    "gpt-6-luna": Model(
        "gpt-6-luna",
        "https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        usd_per_m_input=0.10,
        usd_per_m_output=0.50,
        reasoning_effort="none",
        max_tokens_field="max_completion_tokens",
    ),
}
