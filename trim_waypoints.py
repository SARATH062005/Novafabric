import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Trim waypoints from a JSON file.")
    parser.add_argument("--path", type=str, default="folding_waypoints_1_seq.json", help="Path to the waypoints JSON file")
    parser.add_argument("--num", type=int, default=10, help="Number of points to remove from the end")
    parser.add_argument("--output", type=str, help="Path to save the trimmed file (defaults to overwrite input)")
    args = parser.parse_args()

    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: {file_path} not found.")
        return

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    if not isinstance(data, list):
        print("Error: JSON content is not a list of points.")
        return

    original_length = len(data)
    if args.num >= original_length:
        print(f"Warning: Removing {args.num} points would empty the file. No changes made.")
        return

    # Trim the data
    trimmed_data = data[:-args.num]
    
    save_path = Path(args.output) if args.output else file_path
    
    try:
        with open(save_path, "w") as f:
            json.dump(trimmed_data, f)
        print(f"Successfully trimmed {args.num} points.")
        print(f"Original points: {original_length}")
        print(f"New total: {len(trimmed_data)}")
        print(f"Saved to: {save_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
