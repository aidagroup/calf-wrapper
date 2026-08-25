from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import computed_field

REPO_PATH = Path(__file__).parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_PATH / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    MINIO_PORT: int
    MINIO_CONSOLE_PORT: int
    MLFLOW_PORT: int
    EXPERIMENT_TRACKING_HOST: str

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

    ARTIFACT_UPLOAD_POLL_INTERVAL: float = 30.0
    LOG_ARTEFACTS_UPLOAD_PATH: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def LOG_ARTIFACT_DIR(self) -> Path:
        return REPO_PATH / self.LOG_ARTEFACTS_UPLOAD_PATH

    @computed_field  # type: ignore[prop-decorator]
    @property
    def MLFLOW_S3_ENDPOINT_URL(self) -> str:
        return f"http://{self.EXPERIMENT_TRACKING_HOST}:{self.MINIO_PORT}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return f"http://{self.EXPERIMENT_TRACKING_HOST}:{self.MLFLOW_PORT}"


config = Config()
