import argparse
import datasets
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",type=str)
    parser.add_argument("--output_dir",type=str)

    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.dataset_path)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    output_file_path = output_dir / "dataset.parquet"
    dataset.to_parquet(output_file_path)