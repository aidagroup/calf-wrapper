"""Upload a small artifact batch to MLflow and verify it byte-for-byte."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_verified_artifact_batch(
    source_root: Path,
    *,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
) -> dict:
    """Upload all files below ``source_root`` together and verify hashes remotely."""

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("verified artifact upload requires an active MLflow run")
    source_root = Path(source_root)
    run_id = active_run.info.run_id
    batch_id = uuid.uuid4().hex
    files = []
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and "_batch_manifests" not in path.parts:
            files.append(
                {
                    "path": path.relative_to(source_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not files:
        raise ValueError("artifact batch must contain at least one file")

    manifest = {
        "format": "calf-wrapper-artifact-batch-v1",
        "run_id": run_id,
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_dir = source_root / "_batch_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{batch_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    client = MlflowClient()
    for attempt in range(1, max_retries + 1):
        try:
            client.log_artifacts(run_id, str(source_root))
            with tempfile.TemporaryDirectory(prefix=f"verify_{batch_id}_") as tmp:
                verification_root = Path(tmp)
                for expected in files:
                    downloaded = Path(
                        client.download_artifacts(
                            run_id, expected["path"], dst_path=str(verification_root)
                        )
                    )
                    if downloaded.stat().st_size != expected["size"]:
                        raise RuntimeError(
                            f"remote size mismatch for {expected['path']}"
                        )
                    if _sha256(downloaded) != expected["sha256"]:
                        raise RuntimeError(
                            f"remote SHA-256 mismatch for {expected['path']}"
                        )
                remote_manifest = Path(
                    client.download_artifacts(
                        run_id,
                        f"_batch_manifests/{manifest_path.name}",
                        dst_path=str(verification_root),
                    )
                )
                if remote_manifest.read_bytes() != manifest_path.read_bytes():
                    raise RuntimeError("remote artifact manifest mismatch")
            client.set_tag(run_id, "artifact_upload_status", "verified")
            client.set_tag(run_id, "artifact_batches_verified", "1")
            client.set_tag(run_id, "artifact_last_verified_batch", batch_id)
            return manifest
        except Exception:
            if attempt == max_retries:
                client.set_tag(run_id, "artifact_upload_status", "failed")
                raise
            time.sleep(retry_base_delay * (2 ** (attempt - 1)))

    raise AssertionError("unreachable")
