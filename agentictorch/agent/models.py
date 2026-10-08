"""Models the agent can use: free ones for development, OpenAI only for the final runs."""

from agentictorch.agent.llm import Model

MODELS = {
    # Local and free through Ollama. Greedy decoding can reason until the token cap
    # without answering, so it gets a little temperature and low reasoning effort.
    "ornith-9b": Model(
        "ornith-1.5:9b", "http://localhost:11434/v1", temperature=0.6, reasoning_effort="low"
    ),
    # Free tier through Google's OpenAI-compatible endpoint; its 20 requests a day are
    # too few for an agent run of 30-40 turns.
    "gemini-flash": Model(
        "gemini-3.8-flash",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
        temperature=1.0,
    ),
    # Prices per million tokens as listed on 2026-10-24. Chat Completions takes tool calls
    # only with reasoning effort "none".
    "gpt-6-luna": Model(
        "gpt-6-luna",
        "https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        usd_per_m_input=0.10,
        usd_per_m_output=0.50,
        usd_per_m_cached_input=0.01,
        reasoning_effort="none",
        max_tokens_field="max_completion_tokens",
    ),
    # The stronger model, priced as listed on 2026-10-25. It takes tools with reasoning only
    # through the Responses API.
    "gpt-6.1-sol": Model(
        "gpt-6.1-sol",
        "https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        usd_per_m_input=2.00,
        usd_per_m_output=10.00,
        usd_per_m_cached_input=0.10,
        reasoning_effort="medium",
        api="responses",
    ),
}
