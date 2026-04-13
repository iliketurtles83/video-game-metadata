"""ROM folder parser.

Scans user-provided game folders and returns a normalized inventory table.
This module is intentionally lightweight and focused on inventory extraction:
- infer platform names from folder names via gamelist_folder_mappings.json
- optionally filter by platform-specific extensions via platform_formats.json
- parse game names from filenames
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


DEFAULT_PLATFORM_MAPPINGS_FILE = "utils/gamelist_folder_mappings.json"
DEFAULT_PLATFORM_FORMATS_FILE = "utils/platform_formats.json"


def _load_json_file(path: str | Path) -> dict[str, Any]:
	"""Load a JSON object from file, returning an empty dict on failure."""
	try:
		with open(path, "r", encoding="utf-8") as file:
			data = json.load(file)
	except (OSError, json.JSONDecodeError, TypeError):
		return {}

	return data if isinstance(data, dict) else {}


def load_platform_mappings(
	mappings_file: str = DEFAULT_PLATFORM_MAPPINGS_FILE,
) -> dict[str, str]:
	"""Load platform folder -> canonical platform name mappings."""
	raw = _load_json_file(mappings_file)
	return {
		str(key).strip().lower(): str(value).strip()
		for key, value in raw.items()
		if value is not None
	}


def load_platform_formats(
	formats_file: str = DEFAULT_PLATFORM_FORMATS_FILE,
) -> dict[str, set[str]]:
	"""Load platform -> allowed extension set mappings."""
	raw = _load_json_file(formats_file)
	platform_formats: dict[str, set[str]] = {}

	for platform, extensions in raw.items():
		key = str(platform).strip().lower()
		if not isinstance(extensions, list):
			platform_formats[key] = set()
			continue

		normalized_extensions = {
			str(ext).strip().lower()
			for ext in extensions
			if isinstance(ext, str) and str(ext).strip()
		}
		platform_formats[key] = normalized_extensions

	return platform_formats


def parse_game_name_from_filename(filename: str) -> str:
	"""Parse a readable game name from a ROM filename."""
	stem = Path(filename).stem
	name = stem.replace("_", " ").replace(".", " ")
	name = re.sub(r"\s*[\[(][^\])]*[\])]", "", name)
	name = re.sub(r"\s+", " ", name)
	return name.strip()


def _iter_files(base_path: Path, recursive: bool) -> list[Path]:
	"""Return files under a path, recursively or only top-level."""
	if recursive:
		return [path for path in base_path.rglob("*") if path.is_file()]
	return [path for path in base_path.iterdir() if path.is_file()]


def scan_platform_folder(
	platform_folder: str | Path,
	platform_key: Optional[str] = None,
	mappings_file: str = DEFAULT_PLATFORM_MAPPINGS_FILE,
	formats_file: str = DEFAULT_PLATFORM_FORMATS_FILE,
	strict_extensions: bool = False,
	recursive: bool = True,
) -> pd.DataFrame:
	"""Scan one platform folder and return discovered games.

	Args:
		platform_folder: Folder containing ROM files.
		platform_key: Optional explicit platform key (e.g. "snes").
					  If not provided, folder name is used.
		mappings_file: JSON mapping from folder keys to canonical platform names.
		formats_file: JSON mapping from platform keys to valid file extensions.
		strict_extensions: If True, only keep files with configured extensions.
		recursive: If True, scan subfolders too.

	Returns:
		DataFrame with columns:
		- name
		- filename
		- platform
		- platform_key
		- source_path
		- extension
	"""
	folder_path = Path(platform_folder)
	if not folder_path.exists() or not folder_path.is_dir():
		return pd.DataFrame(columns=["name", "filename", "platform", "platform_key", "source_path", "extension"])

	mappings = load_platform_mappings(mappings_file)
	platform_formats = load_platform_formats(formats_file)

	resolved_platform_key = (platform_key or folder_path.name).strip().lower()
	canonical_platform_name = mappings.get(resolved_platform_key, folder_path.name)
	allowed_extensions = platform_formats.get(resolved_platform_key, set())

	rows: list[dict[str, Any]] = []

	for file_path in _iter_files(folder_path, recursive=recursive):
		extension = file_path.suffix.lower()
		if strict_extensions and allowed_extensions and extension not in allowed_extensions:
			continue

		rows.append(
			{
				"name": parse_game_name_from_filename(file_path.name),
				"filename": file_path.name,
				"platform": canonical_platform_name,
				"platform_key": resolved_platform_key,
				"source_path": str(file_path),
				"extension": extension,
			}
		)

	return pd.DataFrame(rows)


def scan_games_folders(
	games_root: str | Path,
	platform_folders: Optional[Sequence[str | Path]] = None,
	mappings_file: str = DEFAULT_PLATFORM_MAPPINGS_FILE,
	formats_file: str = DEFAULT_PLATFORM_FORMATS_FILE,
	strict_extensions: bool = False,
	recursive: bool = True,
) -> pd.DataFrame:
	"""Scan multiple platform folders and return a single inventory dataframe.

	Args:
		games_root: Root games folder containing platform subfolders.
		platform_folders: Optional explicit list of folders to scan. If omitted,
						  direct child directories of games_root are scanned.
		mappings_file: Path to folder->platform mapping JSON.
		formats_file: Path to platform->extensions mapping JSON.
		strict_extensions: If True, enforce extension filtering per platform.
		recursive: If True, recurse inside each platform folder.
	"""
	root_path = Path(games_root)
	if platform_folders is None:
		if not root_path.exists() or not root_path.is_dir():
			return pd.DataFrame(columns=["name", "filename", "platform", "platform_key", "source_path", "extension"])
		folders_to_scan = [path for path in root_path.iterdir() if path.is_dir()]
	else:
		folders_to_scan = [Path(path) for path in platform_folders]

	frames: list[pd.DataFrame] = []
	for folder in folders_to_scan:
		frame = scan_platform_folder(
			platform_folder=folder,
			platform_key=folder.name,
			mappings_file=mappings_file,
			formats_file=formats_file,
			strict_extensions=strict_extensions,
			recursive=recursive,
		)
		if not frame.empty:
			frames.append(frame)

	if not frames:
		return pd.DataFrame(columns=["name", "filename", "platform", "platform_key", "source_path", "extension"])

	return pd.concat(frames, ignore_index=True)

