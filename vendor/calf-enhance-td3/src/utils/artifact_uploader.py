import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from mlflow.tracking import MlflowClient

from src.config import config


class ArtifactUploadError(RuntimeError):
    """Raised when a staged artifact batch cannot be uploaded and verified."""


class ArtifactUploader:
    """Background worker that batches, uploads, and verifies MLflow artifacts."""

    MANIFEST_DIR = "_batch_manifests"

    def __init__(
        self,
        base_staging_dir: Path,
        poll_interval: float = 30.0,
        *,
        client: MlflowClient | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
    ):
        self.base_staging_dir = Path(base_staging_dir)
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        self._client = client or MlflowClient()
        self._run_id: str | None = None
        self._staging_dir: Path | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._verified_batches = 0

    @property
    def staging_dir(self) -> Path:
        if self._staging_dir is None:
            raise RuntimeError("ArtifactUploader not started")
        return self._staging_dir

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def start(self, run_id: str):
        self._run_id = run_id
        self._staging_dir = self.base_staging_dir / run_id / "staging"
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._error = None
        self._verified_batches = 0
        self._client.set_tag(run_id, "artifact_upload_status", "running")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=False)
        self._thread.start()
        logger.info(f"ArtifactUploader started for run {run_id}")

    def stop(self):
        """Block until every staged artifact is remotely verified."""

        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._error is not None:
            raise ArtifactUploadError(
                f"Artifact uploader failed for run {self._run_id}; "
                f"staged files were preserved under {self.staging_dir.parent}"
            ) from self._error

        self._client.set_tag(self._run_id, "artifact_upload_status", "verified")
        if self._staging_dir:
            run_dir = self._staging_dir.parent
            if run_dir.exists():
                shutil.rmtree(run_dir)
        logger.info(
            f"ArtifactUploader stopped after verifying {self._verified_batches} batch(es)"
        )

    def stage_files(self, source_paths: list[Path], artifact_subdir: str) -> list[Path]:
        """Copy a group of complete files into staging as one atomic producer step."""

        if self._error is not None:
            raise ArtifactUploadError(
                "Cannot stage files after uploader failure"
            ) from self._error
        staged_paths = []
        with self._lock:
            target_dir = self.staging_dir / artifact_subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_path in source_paths:
                source_path = Path(source_path)
                target_path = target_dir / source_path.name
                temporary_path = target_path.with_name(
                    f".{target_path.name}.{uuid.uuid4().hex}.tmp"
                )
                shutil.copy2(source_path, temporary_path)
                os.replace(temporary_path, target_path)
                staged_paths.append(target_path)
        return staged_paths

    def _worker_loop(self):
        try:
            while not self._stop_event.is_set():
                self._upload_staging_batch()
                self._stop_event.wait(self.poll_interval)
            self._flush_remaining()
        except BaseException as error:
            self._error = error
            self._stop_event.set()
            try:
                self._client.set_tag(self._run_id, "artifact_upload_status", "failed")
            except Exception:
                logger.exception("Could not mark artifact uploader as failed")
            logger.exception(f"ArtifactUploader failed: {error}")

    def _flush_remaining(self):
        """Upload existing batch directories, then atomically drain staging."""

        while True:
            pending_batches = sorted(self.staging_dir.parent.glob("batch_*"))
            if pending_batches:
                for batch_dir in pending_batches:
                    self._upload_and_verify(batch_dir, final_flush=True)
                continue

            batch_dir = self._swap_staging()
            if batch_dir is None:
                return
            self._upload_and_verify(batch_dir, final_flush=True)

    def _upload_staging_batch(self):
        batch_dir = self._swap_staging()
        if batch_dir is not None:
            self._upload_and_verify(batch_dir, final_flush=False)

    def _upload_and_verify(self, batch_dir: Path, *, final_flush: bool):
        manifest_path, manifest = self._write_manifest(batch_dir)
        batch_id = manifest["batch_id"]
        file_count = len(manifest["files"])
        operation = "Final flush" if final_flush else "Uploading"

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"{operation} batch {batch_id}: {file_count} artifact(s), "
                f"attempt {attempt}/{self.max_retries}"
            )
            try:
                self._client.log_artifacts(self._run_id, str(batch_dir))
                self._verify_remote_batch(manifest_path, manifest)
                self._verified_batches += 1
                self._client.set_tag(
                    self._run_id,
                    "artifact_batches_verified",
                    str(self._verified_batches),
                )
                self._client.set_tag(
                    self._run_id, "artifact_last_verified_batch", batch_id
                )
                shutil.rmtree(batch_dir)
                logger.info(
                    f"Verified remote artifact batch {batch_id} ({file_count} files)"
                )
                return
            except Exception as error:
                logger.warning(
                    f"Artifact batch {batch_id} attempt {attempt} failed: {error}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

        raise ArtifactUploadError(
            f"Artifact batch {batch_id} failed after {self.max_retries} attempts"
        )

    def _write_manifest(self, batch_dir: Path) -> tuple[Path, dict]:
        batch_id = batch_dir.name.removeprefix("batch_")
        files = []
        for artifact_path in sorted(batch_dir.rglob("*")):
            if not artifact_path.is_file():
                continue
            relative_path = artifact_path.relative_to(batch_dir)
            if relative_path.parts[0] == self.MANIFEST_DIR:
                continue
            files.append(
                {
                    "path": relative_path.as_posix(),
                    "size": artifact_path.stat().st_size,
                    "sha256": self._sha256(artifact_path),
                }
            )

        manifest = {
            "format": "calf-wrapper-artifact-batch-v1",
            "run_id": self._run_id,
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        manifest_dir = batch_dir / self.MANIFEST_DIR
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{batch_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest_path, manifest

    def _verify_remote_batch(self, manifest_path: Path, manifest: dict):
        batch_dir = manifest_path.parent.parent
        with tempfile.TemporaryDirectory(
            prefix=f"verify_{manifest['batch_id']}_", dir=batch_dir.parent
        ) as verification_dir:
            verification_root = Path(verification_dir)
            for expected in manifest["files"]:
                downloaded_path = Path(
                    self._client.download_artifacts(
                        self._run_id,
                        expected["path"],
                        dst_path=str(verification_root),
                    )
                )
                actual_size = downloaded_path.stat().st_size
                actual_sha256 = self._sha256(downloaded_path)
                if actual_size != expected["size"]:
                    raise ArtifactUploadError(
                        f"Remote size mismatch for {expected['path']}: "
                        f"expected {expected['size']}, found {actual_size}"
                    )
                if actual_sha256 != expected["sha256"]:
                    raise ArtifactUploadError(
                        f"Remote SHA-256 mismatch for {expected['path']}"
                    )

            manifest_artifact_path = f"{self.MANIFEST_DIR}/{manifest_path.name}"
            downloaded_manifest = Path(
                self._client.download_artifacts(
                    self._run_id,
                    manifest_artifact_path,
                    dst_path=str(verification_root),
                )
            )
            if downloaded_manifest.read_bytes() != manifest_path.read_bytes():
                raise ArtifactUploadError(
                    f"Remote manifest mismatch for batch {manifest['batch_id']}"
                )

    def _swap_staging(self) -> Path | None:
        """Atomically replace staging and return its immutable batch directory."""

        with self._lock:
            if not self._has_files():
                return None
            batch_dir = self.staging_dir.parent / f"batch_{uuid.uuid4().hex}"
            self.staging_dir.rename(batch_dir)
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            return batch_dir

    def _has_files(self) -> bool:
        return any(artifact.is_file() for artifact in self.staging_dir.rglob("*"))

    @staticmethod
    def _sha256(artifact_path: Path) -> str:
        digest = hashlib.sha256()
        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


_uploader: ArtifactUploader | None = None


def init_artifact_uploader(
    run_id: str,
    base_staging_dir: Path = config.LOG_ARTIFACT_DIR,
    poll_interval: float = 30.0,
):
    global _uploader
    _uploader = ArtifactUploader(base_staging_dir, poll_interval)
    _uploader.start(run_id)


def shutdown_artifact_uploader():
    global _uploader
    if _uploader:
        try:
            _uploader.stop()
        finally:
            _uploader = None


def get_artifact_uploader() -> ArtifactUploader | None:
    return _uploader
