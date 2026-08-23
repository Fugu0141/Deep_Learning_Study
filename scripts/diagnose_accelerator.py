from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_learning_studio.accelerators import accelerator_report, detect_accelerators


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Deep Learning Studio accelerators")
    parser.add_argument(
        "--require",
        choices=["cuda", "rocm", "xpu", "mps", "directml", "cpu"],
        help="Exit with an error unless this backend is available",
    )
    args = parser.parse_args()
    print(accelerator_report())
    if args.require:
        selected = next(item for item in detect_accelerators() if item.key == args.require)
        if not selected.available:
            print(f"\nERROR: {selected.label} is not available: {selected.detail}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
