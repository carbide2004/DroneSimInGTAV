import argparse
import json
import random
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Shuffle the order of data in train_data_all.json")
    parser.add_argument(
        "--input_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output_json",
        default=str(_repo_root() / "dataset" / "train_data_all_shuffled.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of creating a new one"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Read input data
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Reading data from: {input_path}")
    data = _read_json(input_path)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list")
    
    original_count = len(data)
    print(f"Original data count: {original_count}")
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Determine output path
    if args.inplace:
        output_path = input_path
    else:
        output_path = Path(args.output_json)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write shuffled data
    print(f"Writing shuffled data to: {output_path}")
    _write_json(output_path, shuffled_data)
    
    print(f"Successfully shuffled {original_count} entries with seed {args.seed}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())