import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(prog="ghostlab", description="GhostLab local-first analytics + ML pipelines")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="Check environment + folders")

    args = p.parse_args()

    if args.cmd == "doctor":
        root = Path(__file__).resolve().parents[2]
        print("OK: ghostlab CLI is wired up.")
        print(f"Repo root: {root}")
        return

    p.print_help()

if __name__ == "__main__":
    main()
