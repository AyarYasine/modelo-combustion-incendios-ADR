"""
Helper para cargar variables de entorno de .env.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

def init_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # fallback: load default .env if exists
        load_dotenv()

    # Aviso si faltan credenciales de Plotly
    usr = os.getenv("PLOTLY_USERNAME")
    key = os.getenv("PLOTLY_API_KEY")
    if not usr or not key:
        print("[AVISO] Sin credenciales de Plotly configuradas. Las gráficas se guardarán en local.")
    return usr, key
