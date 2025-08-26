import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "storage" / "chroma"))
DEFAULT_DOCS_DIR = os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "uploads"))

# Models
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Misc
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

Path(DEFAULT_CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(DEFAULT_DOCS_DIR).mkdir(parents=True, exist_ok=True)
