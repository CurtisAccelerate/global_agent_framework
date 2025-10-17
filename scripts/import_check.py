import importlib
import os
import sys


def main() -> None:
    # Ensure project root is on sys.path when running from scripts/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    modules = [
        "agent",
        "pipeline",
        "agent_pipeline_declarations",
        "tools.polymarket_gamma",
        "api",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()


