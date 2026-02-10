from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # auth
    credentials_path: Path = Path("credentials.json")
    token_path: Path = Path("token.json")

    # storage
    db_path: Path = Path("classroom_agent.sqlite")
    cache_dir: Path = Path("cache_files")

    # ingestion & caching
    max_pages_or_slides: int = 100
    max_summarize_per_run: int = 10

    # summarization
    max_chars: int = 12000
    overlap: int = 800
    max_chunks: int = 6
    max_completion_tokens: int = 1500

    # retrieval
    top_k_retrieval: int = 5

    # digest
    lookback_hours_for_digest: int = 12

    # embeddings
    use_embeddings: bool = True
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # llm
    nemotron_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

    # runtime
    poll_hours: float = 3.5
    allow_hitl: bool = False
