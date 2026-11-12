#!/usr/bin/env python3
"""Check the existence, identity, and optional Git inclusion of indexed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "evidence-manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--require-tracked", action="store_true",
        help="Also require the manifest, checker, and evidence files in Git's index.",
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("unsupported manifest schema")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("files must be a nonempty list")
        tracked = None
        if args.require_tracked:
            result = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "--cached", "-z"],
                check=True, capture_output=True,
            )
            tracked = set(result.stdout.decode("utf-8").split("\0"))
        failures: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each file entry must be an object")
            name, expected = entry.get("path"), entry.get("sha256")
            if not isinstance(name, str) or not name or "\\" in name or ":" in name:
                raise ValueError("file paths must be relative POSIX paths")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or name != relative.as_posix():
                raise ValueError("file paths must be normalized and inside the repository")
            if name in seen:
                raise ValueError(f"duplicate path: {name}")
            seen.add(name)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"invalid SHA-256: {name}")
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError(f"file resolves outside the repository: {name}")
            if not path.is_file():
                failures.append(f"MISSING {name}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                failures.append(f"CHANGED {name}")
            if tracked is not None and name not in tracked:
                failures.append(f"UNTRACKED {name}")
        if tracked is not None:
            for own_path in [args.manifest.resolve(), Path(__file__).resolve()]:
                if not own_path.is_relative_to(ROOT):
                    failures.append("PUBLICATION manifest/checker is outside the repository")
                elif own_path.relative_to(ROOT).as_posix() not in tracked and own_path.relative_to(ROOT).as_posix() not in seen:
                    failures.append(f"UNTRACKED {own_path.relative_to(ROOT).as_posix()}")
        if failures:
            for failure in failures:
                print(failure)
            print(f"FAIL: {len(failures)} issue(s) across {len(entries)} evidence files.")
            return 1
        print(f"PASS: {len(entries)} evidence files match the manifest.")
        if tracked is not None:
            print("Indexed evidence, manifest, and checker are present in Git's index.")
        print("This checks file identity, not scientific validity or full dependency closure.")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
