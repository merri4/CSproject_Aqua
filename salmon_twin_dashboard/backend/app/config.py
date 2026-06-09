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

    # Ollama endpoint and model. Change the model name to your local model tag.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4"

    # Optional: if your RAG pipeline exposes an HTTP endpoint, set it here.
    # It should accept POST JSON {"query": "..."} and return either {"context": "..."}
    # or any JSON/text payload that can be converted into context.
    RAG_ENDPOINT: str | None = None

    # Frontend origin for local dev.
    CORS_ORIGIN: str = "http://localhost:5173"

    class Config:
        env_file = ".env"


settings = Settings()
