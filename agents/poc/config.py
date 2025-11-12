from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl
from typing import Annotated

class Settings(BaseSettings):
    # Define the application name with a default value
    app_name: str = "Generative AI Services"
    azure_endpoint_url:     Annotated[HttpUrl, Field(alias='azure_endpoint_url')]
    azure_deployment_name:  Annotated[str, Field(min_length=5, default='gpt-5-nano')]
    azure_openai_api_key:   Annotated[str, Field(min_length=5)]
    azure_openai_version:   Annotated[str, Field(min_length=5, default='2024-12-01-preview')]

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding ="utf-8"
    )
    #class Config:
    #    # Specify the name of the environment file and enconding to load variables from
    #    env_file = ".env"
    #    env_file_encoding = "utf-8"

settings = Settings()
