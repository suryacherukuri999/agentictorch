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
}
