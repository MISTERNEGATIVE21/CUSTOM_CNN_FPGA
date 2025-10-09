#!/usr/bin/env python3
"""Simple interactive launcher for project scripts."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def discover_scripts(directory: Path) -> list[Path]:
    ignore = {"__pycache__", ".venv", "venv", ".git"}
    scripts: list[Path] = []
    for path in directory.rglob("*.py"):
        if any(part in ignore for part in path.parts):
            continue
        if path.name == "script_launcher.py":
            continue
        scripts.append(path.relative_to(directory))
    return sorted(scripts)


def run_script(script: Path) -> int:
    cmd = [sys.executable, str(script)]
    print(f"\n▶ Running: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        return 130


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple interactive launcher for project scripts.")
    parser.add_argument("--list", action="store_true", help="List available scripts and exit")
    parser.add_argument("--run", type=str, help="Run a script by index (1-based) or by filename")
    args = parser.parse_args()

    scripts = discover_scripts(PROJECT_ROOT)
    if not scripts:
        print("No Python scripts found.")
        return

    def print_scripts() -> None:
        print("\n=== Python Script Launcher ===")
        for idx, script in enumerate(scripts, start=1):
            print(f"{idx:2} → {script}")
        print(" 0 → Exit")

    # Non-interactive options: --list or --run, or env var SCRIPT_RUN
    env_run = os.environ.get("SCRIPT_RUN")
    if args.list:
        print_scripts()
        return
    if args.run or env_run:
        target = args.run if args.run else env_run
        # numeric index
        if target.isdigit():
            index = int(target) - 1
            if index < 0 or index >= len(scripts):
                print(f"Invalid selection: {target}")
                sys.exit(2)
            selected = scripts[index]
        else:
            # try to find by name (suffix match)
            matches = [s for s in scripts if str(s).endswith(target) or s.name == target]
            if not matches:
                print(f"No script matching '{target}'")
                sys.exit(3)
            selected = matches[0]
        code = run_script(selected)
        print(f"\n⇦ Script exited with status {code}\n")
        sys.exit(code)

    # Interactive mode: if stdin is not a tty, avoid input() which raises EOFError
    if not sys.stdin.isatty():
        print("Non-interactive session detected. Use --list or --run to operate this launcher in scripts/CI.")
        print_scripts()
        return

    while True:
        print_scripts()
        try:
            choice = input("Select a script to run: ").strip()
        except EOFError:
            print("\nEOF received; exiting.")
            break

        if choice == "0":
            print("Goodbye!")
            break
        if not choice.isdigit():
            print("Please enter a number.")
            continue

        index = int(choice) - 1
        if index < 0 or index >= len(scripts):
            print("Invalid selection.")
            continue

        selected = scripts[index]
        code = run_script(selected)
        print(f"\n⇦ Script exited with status {code}\n")


if __name__ == "__main__":
    main()
