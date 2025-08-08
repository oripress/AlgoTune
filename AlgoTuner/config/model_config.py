from typing import Optional
from pydantic import BaseModel, SecretStr


class GenericAPIModelConfig(BaseModel):
    name: str
    api_key: SecretStr  # Use SecretStr for sensitive data
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: int = 4096
    spend_limit: float = 0.0
    api_key_env: str  # Environment variable name for the API key


class GlobalConfig(BaseModel):
    spend_limit: float = 0.5
    total_messages: int = 9999
    max_messages_in_history: int = 5
    oracle_time_limit: int = 100  # in milliseconds
