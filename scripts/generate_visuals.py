#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vol_surface_hedging_lab.visualisation import generate_visual_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SVG visual artefacts from JSON result files."
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory containing result JSON files")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    surface = _read_json(out / "surface_results.json")
    model = _read_json(out / "model_comparison_results.json")
    hedging = _read_json(out / "hedging_backtest_results.json")
    rolling = _read_json(out / "rolling_recalibration_results.json")

    generated = generate_visual_artifacts(
        surface_result=surface,
        model_result=model,
        hedging_result=hedging,
        rolling_result=rolling,
        output_dir=out,
    )
    print(f"Generated {len(generated)} visual files in {out}")


if __name__ == "__main__":
    main()
