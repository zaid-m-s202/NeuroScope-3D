"""
NeuroScope 3D - Ephemeral Case Storage & Lifecycle Service
Manages private temporary case files, SHA-256 checksums, and 24-hour automatic retention purge.
Adheres strictly to PRD_BASE.md (FR-9) and ARCHITECTURE.md.
"""
from typing import Dict, Any, Optional, Tuple, List
import os
import re
import shutil
import time
import uuid
import json
import hashlib

STORAGE_BASE_DIR = os.getenv("STORAGE_DIR", "./storage/cases")
RETENTION_HOURS = int(os.getenv("TEMP_RETENTION_HOURS", "24"))


VALID_CASE_ID_PATTERN = re.compile(r"^(case_|BraTS)[a-zA-Z0-9_\-]+$")


def compute_file_sha256(file_path: str) -> str:
    """Computes hexadecimal SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_case_dir(case_id: str) -> str:
    """
    Securely resolves and returns the absolute canonical path to a case's private storage directory.
    Enforces strict identifier syntax and verifies containment within STORAGE_BASE_DIR.
    Guarantees protection against path traversal, absolute path injection, symlink escapes, and null byte injection.
    """
    if not isinstance(case_id, str):
        raise ValueError("Invalid case identifier: must be a string.")

    # Reject unexpected lengths
    if len(case_id) < 6 or len(case_id) > 80:
        raise ValueError("Invalid case identifier: invalid length.")

    # Explicitly reject dangerous traversal characters, separators, colons, null bytes, percent signs
    if any(c in case_id for c in ("/", "\\", "..", "\x00", "%", "\n", "\r", ":")):
        raise ValueError("Path traversal detected for case identifier.")

    # Reject identifiers not adhering to case_<alphanumeric/hyphen/underscore>
    if not VALID_CASE_ID_PATTERN.match(case_id):
        raise ValueError("Invalid case identifier: malformed syntax.")

    canonical_base = os.path.realpath(os.path.abspath(STORAGE_BASE_DIR))
    raw_case_path = os.path.abspath(os.path.join(canonical_base, case_id))
    canonical_case_dir = os.path.realpath(raw_case_path)

    # Verify canonical path containment within canonical base directory
    try:
        common = os.path.commonpath([canonical_base, canonical_case_dir])
    except ValueError:
        # Handles Windows cross-drive path injection attempts
        raise ValueError("Path traversal detected: cross-drive path resolution attempted.")

    if common != canonical_base:
        raise ValueError("Path traversal detected for case identifier: escapes storage root.")

    # Disallow symlinks pointing outside the storage base
    if os.path.islink(raw_case_path):
        link_target = os.path.realpath(raw_case_path)
        try:
            link_common = os.path.commonpath([canonical_base, link_target])
            if link_common != canonical_base:
                raise ValueError("Path traversal detected: symlink points outside storage root.")
        except ValueError:
            raise ValueError("Path traversal detected: symlink cross-drive reference.")

    return raw_case_path


def validate_case_id(case_id: str) -> bool:
    """Returns True if case_id is syntactically valid and stays within storage root."""
    try:
        get_case_dir(case_id)
        return True
    except (ValueError, TypeError):
        return False


def create_case_workspace() -> Tuple[str, str]:
    """
    Creates a unique private directory for an uploaded case.
    Returns (case_id, case_dir).
    """
    case_id = f"case_{uuid.uuid4()}"
    case_dir = get_case_dir(case_id)
    os.makedirs(case_dir, exist_ok=True)
    return case_id, case_dir


def save_case_metadata(case_id: str, metadata: Dict[str, Any]) -> None:
    """Saves state and results to case_state.json inside private case directory."""
    case_dir = get_case_dir(case_id)
    if os.path.exists(case_dir):
        state_file = os.path.join(case_dir, "case_state.json")
        with open(state_file, "w") as f:
            json.dump(metadata, f, indent=2)


def load_case_metadata(case_id: str) -> Optional[Dict[str, Any]]:
    """Loads state and results from case_state.json if available."""
    try:
        case_dir = get_case_dir(case_id)
    except ValueError:
        return None
    state_file = os.path.join(case_dir, "case_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return None


def cleanup_expired_cases(max_age_hours: int = RETENTION_HOURS) -> int:
    """
    Automatically purges temporary case directories older than max_age_hours (default: 24h).
    Ensures that only directories strictly inside canonical storage root are removed,
    and never follows symlinks outside the storage root.
    Returns count of cleaned cases.
    """
    canonical_base = os.path.realpath(os.path.abspath(STORAGE_BASE_DIR))
    if not os.path.exists(canonical_base):
        return 0

    now = time.time()
    cutoff_seconds = max_age_hours * 3600
    cleaned_count = 0

    try:
        entries = os.listdir(canonical_base)
    except OSError:
        return 0

    for item in entries:
        if item in (".", "..") or not isinstance(item, str):
            continue

        item_path = os.path.join(canonical_base, item)

        # Ensure item remains strictly inside canonical_base
        try:
            real_item_path = os.path.realpath(item_path)
            if os.path.commonpath([canonical_base, real_item_path]) != canonical_base:
                continue
        except (ValueError, OSError):
            continue

        if os.path.islink(item_path):
            # If item is a symlink, safely remove the link itself, never delete external target
            try:
                os.unlink(item_path)
                cleaned_count += 1
            except OSError:
                pass
            continue

        if os.path.isdir(item_path):
            try:
                dir_mtime = os.path.getmtime(item_path)
                if (now - dir_mtime) > cutoff_seconds:
                    shutil.rmtree(item_path)
                    cleaned_count += 1
            except OSError:
                pass

    return cleaned_count


def list_active_cases() -> List[Dict[str, Any]]:
    """
    Lists active case sessions in storage, returning their summary metadata.
    Sorted by last modified time (newest first).
    """
    canonical_base = os.path.realpath(os.path.abspath(STORAGE_BASE_DIR))
    if not os.path.exists(canonical_base):
        return []

    results = []
    try:
        entries = os.listdir(canonical_base)
    except OSError:
        return []

    for item in entries:
        if not item.startswith("case_"):
            continue
        case_dir = os.path.join(canonical_base, item)
        if not os.path.isdir(case_dir):
            continue

        state_file = os.path.join(case_dir, "case_state.json")
        mtime = os.path.getmtime(case_dir)
        has_mask = os.path.exists(os.path.join(case_dir, "predicted_seg.nii.gz")) or os.path.exists(os.path.join(case_dir, "predicted_mask.nii.gz"))
        files_in_dir = os.listdir(case_dir) if os.path.exists(case_dir) else []
        has_expert = os.path.exists(os.path.join(case_dir, "expert_mask.nii.gz")) or any("expert" in f.lower() for f in files_in_dir)

        info = {
            "case_id": item,
            "modified_timestamp": mtime,
            "status": "completed" if has_mask else "ready",
            "has_mask": has_mask,
            "has_expert_mask": has_expert,
        }
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    meta = json.load(f)
                    info["status"] = meta.get("status") or ("completed" if has_mask else "ready")
                    info["validation_status"] = meta.get("validation_status", "passed")
                    info["spatial_summary"] = meta.get("spatial_summary") or meta.get("spatial_metadata")
                    info["volumes"] = meta.get("volumes")
            except Exception:
                pass
        results.append(info)

    results.sort(key=lambda x: x["modified_timestamp"], reverse=True)
    return results
