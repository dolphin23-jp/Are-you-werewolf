from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    werewolf_env: str = "development"
    werewolf_cors_origins: str = "http://localhost:5173"
    werewolf_log_level: str = "INFO"
    werewolf_session_store: str = "memory"
    werewolf_rng_seed: int | None = None

    werewolf_llm_provider: str = "mock"

    luna_api_key: str = ""
    luna_base_url: str = "https://api.example.com/v1"
    luna_model: str = "gpt-5.6-luna"
    luna_max_concurrency: int = 6
    luna_timeout_seconds: float = 30.0
    luna_max_retries: int = 2

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.werewolf_cors_origins.split(",") if o.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
