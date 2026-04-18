from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.evaluation.runner import run_corpus_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AST corpus evaluation on Test_Images.")
    parser.add_argument(
        "--test-root",
        default=str(ROOT.parent / "Test_Images"),
        help="Path to the Test_Images root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT.parent / "evaluation" / "runs"),
        help="Directory where evaluation runs will be written.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source filters such as custom_plates, dryad_sirscan, custom_plate, dryad_original.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional image limit for quick checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat review-required rows as failures in the aggregate pass/fail summary.",
    )
    args = parser.parse_args()

    outcome = run_corpus_evaluation(
        test_root=Path(args.test_root),
        output_root=Path(args.output_root),
        sources=set(args.sources) if args.sources else None,
        limit=args.limit,
        strict_mode=args.strict,
    )
    print(json.dumps(outcome, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

