import os
from dotenv import load_dotenv

load_dotenv()  # This loads the environment variables from .env.

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
