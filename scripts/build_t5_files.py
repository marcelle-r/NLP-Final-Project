#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import argparse


def build_t5_file(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    # We assume the CSV has at least:
    # - 'name'  → recipe title
    # - 'ingredients'
    # - 'steps'
    # This matches Food.com official schema.

    def make_prompt(row):
        return (
            f"Generate a diabetes-friendly recipe for: {row['name']}\n"
            f"Include ingredients and steps.\n"
        )

    def make_target(row):
        return f"Ingredients:\n{row['ingredients']}\n\nSteps:\n{row['steps']}"

    out = pd.DataFrame({
        "prompt": df.apply(make_prompt, axis=1),
        "target": df.apply(make_target, axis=1),
    })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved T5 training file to {output_path} with {len(out)} rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    build_t5_file(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
