#!/usr/bin/env python3

import re
from pathlib import Path
from typing import Dict, Optional

import yaml


DEFAULT_PACKAGE_MAPPING_PATH = Path(__file__).resolve().parent.parent / "config" / "package_mapping.yaml"


def load_package_mapping(path: Optional[str] = None) -> Dict[str, str]:
    mapping_path = Path(path) if path else DEFAULT_PACKAGE_MAPPING_PATH
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    return mapping or {}


_PACKAGE_MAPPING = load_package_mapping()


def resolve_distribution_name(import_name: str) -> Optional[str]:
    return _PACKAGE_MAPPING.get(import_name)


def normalize_distribution_name(distribution_name: str) -> str:
    return re.sub(r"[-_.]+", "-", distribution_name).lower()
