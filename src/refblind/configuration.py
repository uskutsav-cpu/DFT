"""Configuration loading with duplicate-key rejection for YAML and JSON."""

from __future__ import annotations

from pathlib import Path

import yaml

from .provenance import digest, read_json


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = read_json(path)
    else:
        try:
            value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    digest(value)  # Reject NaN/Infinity and non-JSON-compatible YAML objects.
    forbidden = {"token", "password", "api_key", "secret", "access_token"}

    def check(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if str(key).lower() in forbidden:
                    raise ValueError("do not store credentials in reproducible configuration files")
                check(item)
        elif isinstance(obj, list):
            for item in obj:
                check(item)

    check(value)
    return value
