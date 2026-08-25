import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MINIO_PORT", "9030")
os.environ.setdefault("MINIO_CONSOLE_PORT", "9031")
os.environ.setdefault("MLFLOW_PORT", "5001")
os.environ.setdefault("EXPERIMENT_TRACKING_HOST", "127.0.0.1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("LOG_ARTEFACTS_UPLOAD_PATH", "/tmp/calf-artifact-tests")

from src.utils.artifact_uploader import ArtifactUploadError, ArtifactUploader


class FakeMlflowClient:
    def __init__(self, remote_root: Path, fail_uploads: int = 0):
        self.remote_root = remote_root
        self.fail_uploads = fail_uploads
        self.log_calls = 0
        self.tags = {}

    def set_tag(self, run_id, key, value):
        self.tags[(run_id, key)] = value

    def log_artifacts(self, run_id, local_dir):
        self.log_calls += 1
        if self.log_calls <= self.fail_uploads:
            raise RuntimeError("injected upload failure")
        shutil.copytree(local_dir, self.remote_root / run_id, dirs_exist_ok=True)

    def download_artifacts(self, run_id, artifact_path, dst_path):
        source = self.remote_root / run_id / artifact_path
        destination = Path(dst_path) / artifact_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)


class CorruptingMlflowClient(FakeMlflowClient):
    def download_artifacts(self, run_id, artifact_path, dst_path):
        destination = Path(super().download_artifacts(run_id, artifact_path, dst_path))
        if not artifact_path.startswith(ArtifactUploader.MANIFEST_DIR):
            destination.write_bytes(b"corrupted")
        return str(destination)


def _stage_checkpoint(uploader: ArtifactUploader, source_dir: Path):
    checkpoint = source_dir / "td3_checkpoint_30000_steps.pt"
    metadata = source_dir / "td3_checkpoint_30000_steps.json"
    checkpoint.write_bytes(b"checkpoint bytes")
    metadata.write_text('{"sha256": "example"}\n')
    uploader.stage_files([checkpoint, metadata], "checkpoints")


def test_uploader_batches_and_verifies_every_staged_file(tmp_path):
    client = FakeMlflowClient(tmp_path / "remote")
    uploader = ArtifactUploader(
        tmp_path / "staging",
        poll_interval=3600,
        client=client,
        retry_base_delay=0,
    )
    uploader.start("run-1")
    _stage_checkpoint(uploader, tmp_path)
    uploader.stop()

    remote_run = tmp_path / "remote" / "run-1"
    assert (remote_run / "checkpoints/td3_checkpoint_30000_steps.pt").exists()
    assert (remote_run / "checkpoints/td3_checkpoint_30000_steps.json").exists()
    assert len(list((remote_run / ArtifactUploader.MANIFEST_DIR).glob("*.json"))) == 1
    assert client.tags[("run-1", "artifact_batches_verified")] == "1"
    assert client.tags[("run-1", "artifact_upload_status")] == "verified"


def test_uploader_retries_the_same_batch_without_losing_files(tmp_path):
    client = FakeMlflowClient(tmp_path / "remote", fail_uploads=1)
    uploader = ArtifactUploader(
        tmp_path / "staging",
        poll_interval=3600,
        client=client,
        max_retries=2,
        retry_base_delay=0,
    )
    uploader.start("run-2")
    _stage_checkpoint(uploader, tmp_path)
    uploader.stop()

    assert client.log_calls == 2
    assert (
        tmp_path / "remote/run-2/checkpoints/td3_checkpoint_30000_steps.pt"
    ).exists()


def test_uploader_preserves_failed_batch_when_remote_verification_fails(tmp_path):
    client = CorruptingMlflowClient(tmp_path / "remote")
    uploader = ArtifactUploader(
        tmp_path / "staging",
        poll_interval=3600,
        client=client,
        max_retries=1,
        retry_base_delay=0,
    )
    uploader.start("run-3")
    _stage_checkpoint(uploader, tmp_path)

    with pytest.raises(ArtifactUploadError):
        uploader.stop()

    run_staging = tmp_path / "staging/run-3"
    assert list(run_staging.glob("batch_*"))
    assert client.tags[("run-3", "artifact_upload_status")] == "failed"
