from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "evaluation"
CHROMA_DIR_OPENAI = DATA_DIR / "chroma_db"
CHROMA_DIR_HF = DATA_DIR / "chroma_db_hf"
CHROMA_DIR_NOMIC = DATA_DIR / "chroma_db_nomic"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")