"""
ModelRegistry — reads ml/models/registry.json and exposes typed model entries.

This module is intentionally stateless; it re-reads the JSON on every call to
load_registry() so hot-reloads are safe during development. The ModelService
calls it once at startup and caches the result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Typed representation of one entry in registry.json."""

    model_id: str
    display_name: str
    algorithm_family: str
    artifact_filename: str
    is_primary: bool
    is_active: bool
    supports_probability: bool
    feature_names: list[str]
    processed_feature_count: int
    model_classes: list[int]
    scikit_learn_version: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def load_registry(registry_path: Path) -> list[RegistryEntry]:
    """
    Parse registry.json and return a list of RegistryEntry objects.

    Only active entries (``"is_active": true``) are returned.

    Raises
    ------
    FileNotFoundError
        When registry.json does not exist on disk.
    ValueError
        When the JSON does not conform to the expected schema.
    """
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found at: {registry_path}"
        )

    try:
        raw: dict[str, Any] = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"registry.json is not valid JSON: {exc}") from exc

    if "models" not in raw or not isinstance(raw["models"], list):
        raise ValueError("registry.json must contain a 'models' list.")

    entries: list[RegistryEntry] = []
    for item in raw["models"]:
        if not item.get("is_active", True):
            continue
        try:
            entries.append(
                RegistryEntry(
                    model_id=item["model_id"],
                    display_name=item["display_name"],
                    algorithm_family=item["algorithm_family"],
                    artifact_filename=item["artifact_filename"],
                    is_primary=item.get("is_primary", False),
                    is_active=item.get("is_active", True),
                    supports_probability=item.get("supports_probability", False),
                    feature_names=item.get("feature_names", []),
                    processed_feature_count=item.get("processed_feature_count", 28),
                    model_classes=[int(c) for c in item.get("model_classes", [0, 1])],
                    scikit_learn_version=item.get("scikit_learn_version", "unknown"),
                    notes=item.get("notes", ""),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"registry.json entry missing required key: {exc}"
            ) from exc

    if not entries:
        logger.warning("registry.json contains no active model entries.")

    return entries


def get_primary_model_id(registry_path: Path) -> str:
    """Return the primary_model_id declared at the top of registry.json."""
    if not registry_path.exists():
        return "logistic_regression"
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    return raw.get("primary_model_id", "logistic_regression")
