import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    SOLVER_1_MODEL = "groq/llama-3.3-70b-versatile"   # strong general reasoning
    SOLVER_2_MODEL = "groq/llama-3.1-8b-instant"        # fast, different "voice" than solver 1
    SOLVER_3_MODEL = "groq/openai/gpt-oss-120b"          # OpenAI-flavored reasoning, distinct family

    JUDGE_MODEL = "groq/openai/gpt-oss-20b"               # deliberately NOT the same model as any solver