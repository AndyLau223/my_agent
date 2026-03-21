from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    tavily_api_key: str = ""
    openai_model: str = "gpt-4o"
    sqlite_db_path: str = "./agent_state.db"
    max_critic_iterations: int = 3


settings = Settings()
