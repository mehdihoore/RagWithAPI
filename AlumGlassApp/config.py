from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    ASTRA_DB_TOKEN: str
    ASTRA_DB_API_ENDPOINT: str
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    GOOGLE_CSE_ID: str
    SEARXNG_BASE_URL: str

    model_config = SettingsConfigDict(env_file=".env")
