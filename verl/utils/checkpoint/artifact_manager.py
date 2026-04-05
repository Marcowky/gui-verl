import json
import os
import shutil
from collections.abc import Callable
from typing import Optional


ACTOR_ARTIFACT_METADATA_FILENAME = "artifact_metadata.json"

def get_actor_dir(step_dir: str) -> str:
    return os.path.join(step_dir, "actor")


def get_actor_artifact_metadata_path(actor_dir: str) -> str:
    return os.path.join(actor_dir, ACTOR_ARTIFACT_METADATA_FILENAME)


def write_actor_artifact_metadata(actor_dir: str, kind: str, backend: Optional[str] = None) -> None:
    os.makedirs(actor_dir, exist_ok=True)
    payload = {"kind": kind}
    if backend:
        payload["backend"] = backend
    with open(get_actor_artifact_metadata_path(actor_dir), "w") as f:
        json.dump(payload, f, indent=2)


def read_actor_artifact_metadata(actor_dir: str) -> Optional[dict]:
    metadata_path = get_actor_artifact_metadata_path(actor_dir)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path) as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass

    if not os.path.isdir(actor_dir):
        return None

    names = set(os.listdir(actor_dir))
    if "fsdp_config.json" in names or any(name.startswith("model_world_size_") for name in names):
        return {"kind": "checkpoint", "backend": "fsdp"}

    if (
        "config.json" in names
        or "model.safetensors" in names
        or "model.safetensors.index.json" in names
        or any(name.endswith(".safetensors") for name in names)
    ):
        return {"kind": "weight"}

    return None


def get_actor_artifact_kind(actor_dir: str) -> Optional[str]:
    metadata = read_actor_artifact_metadata(actor_dir)
    if metadata is None:
        return None
    return metadata.get("kind")

def export_actor_weight_artifact(
    source_actor_dir: str,
    target_actor_dir: str,
    converter: Callable[[str, str], None],
    backend: Optional[str] = None,
) -> None:
    if os.path.exists(target_actor_dir):
        shutil.rmtree(target_actor_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(target_actor_dir), exist_ok=True)

    source_kind = get_actor_artifact_kind(source_actor_dir)
    if source_kind == "weight":
        shutil.copytree(source_actor_dir, target_actor_dir)
        write_actor_artifact_metadata(target_actor_dir, kind="weight", backend=backend)
        return

    converter(source_actor_dir, target_actor_dir)
    write_actor_artifact_metadata(target_actor_dir, kind="weight", backend=backend)
