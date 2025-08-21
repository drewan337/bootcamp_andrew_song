from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

def get_key():
    return os.getenv("API_KEY")