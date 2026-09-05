from __future__ import annotations

import platform
import shutil
import sys


def _ok(label: str, value: str) -> None:
    print(f"[OK] {label}: {value}")


def _warn(label: str, value: str) -> None:
    print(f"[WARN] {label}: {value}")


def main() -> int:
    _ok("OS", platform.platform())
    _ok("Python", sys.version.split()[0])

    if sys.version_info < (3, 11):
        _warn("Python version", "3.11+ is recommended")

    tools = ["git", "ollama", "docker"]
    missing = []
    for tool in tools:
        path = shutil.which(tool)
        if path:
            _ok(tool, path)
        else:
            _warn(tool, "not found in PATH")
            missing.append(tool)

    if missing:
        print("\nEnvironment check completed with warnings.")
        print("Missing tools:", ", ".join(missing))
    else:
        print("\nEnvironment check completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
