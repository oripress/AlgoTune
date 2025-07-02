from pydantic import BaseModel, SecretStr


class GenericAPIModelConfig(BaseModel):
    name: str
    api_key: SecretStr  # Use SecretStr for sensitive data
    temperature: float = 0.9
    top_p: float = 1.0
    max_tokens: int = 4096
    spend_limit: float = 0.0
    api_key_env: str  # Environment variable name for the API key


class GlobalConfig(BaseModel):
    spend_limit: float = 0.5
    total_messages: int = 9999
    max_messages_in_history: int = 5
    oracle_time_limit: int = 100  # in milliseconds
