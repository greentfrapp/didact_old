from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    GEMINI_API_KEY: str = ""

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
