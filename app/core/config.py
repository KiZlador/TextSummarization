from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    #MODEL_NAME: str = "cointegrated/rut5-base-absum"
    MODEL_NAME: str = "./notebooks/mt5-small-gazeta-finetuned-lora"
    DEVICE: str = "cuda"
    
    MAX_INPUT_TOKENS: int = 512   
    MAX_OUTPUT_TOKENS: int = 150
    REPETITION_PENALTY: float = 1.3
    NO_REPEAT_NGRAM_SIZE: int = 3
    LENGTH_PENALTY: float = 1.0
    NUM_BEAMS: int = 4
    
    MAX_CHUNK_TOKENS: int = 400
    CHUNK_OVERLAP: int = 50
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()