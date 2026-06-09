from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Example:
    # sqlite:///./sensor.db
    # postgresql+psycopg://user:password@localhost:5432/salmon
    DATABASE_URL: str = "sqlite:///./sensor.db"

    # Table and column names are configurable so you can bind this dashboard to your existing DB.
    SENSOR_TABLE: str = "sensor_readings"
    TIME_COLUMN: str = "timestamp"

    # Comma-separated list of the six controllable sensor/operation variables.
    SENSOR_COLUMNS: str = "temperature,do,ph,ammonia,salinity,turbidity"

    # Optional CSV bootstrap source. When DATABASE_URL points at SQLite and the DB
    # file/table is missing, the backend imports this CSV into SENSOR_TABLE.
    SENSOR_CSV_PATH: str | None = None
    RECENT_FALLBACK_TO_LATEST: bool = True

    # Ollama endpoint and model. Change the model name to your local model tag.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # Optional: if your RAG pipeline exposes an HTTP endpoint, set it here.
    # It should accept POST JSON {"query": "..."} and return either {"context": "..."}
    # or any JSON/text payload that can be converted into context.
    RAG_ENDPOINT: str | None = None
    RAG_ROOT: str | None = None
    RAG_DB_DIR: str | None = None
    RAG_COLLECTION: str = "salmon_farm_manual"
    RAG_MANUALS_PATH: str | None = None
    RAG_TOP_K: int = 3

    # Optional Omniverse control bridge. Confirmed proposals are always written to
    # control_commands; set this to also POST the payload to a controller process.
    OMNIVERSE_CONTROL_ENDPOINT: str | None = None

    # Frontend origin for local dev.
    CORS_ORIGIN: str = "http://localhost:5173"

    class Config:
        env_file = ".env"


settings = Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dashboard_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | None, *fallbacks: str) -> Path | None:
    candidates = [path_value, *fallbacks]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.exists():
            return path
    return None
