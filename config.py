import os
from dotenv import load_dotenv

load_dotenv()

# Embedding
HF_TOKEN = os.getenv("HF_TOKEN")

# Generation
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Storage
CHROMA_PATH = ".chroma_db"