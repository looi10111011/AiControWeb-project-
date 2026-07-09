from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")


    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    primary_llm_provider: str = "anthropic"
    fallback_llm_provider: str = "gemini"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "manuals"

    browser_headless: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
