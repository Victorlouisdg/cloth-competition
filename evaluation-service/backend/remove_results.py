import argparse

from loguru import logger


def remove_masks(dataset_directory: str):
    import os

    for root, _, files in os.walk(dataset_directory):
        for file in files:
            if file == "mask.png" or file == "result.json":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                logger.info(f"Deleted: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively delete all mask.png files in a dataset directory.")
    parser.add_argument("dataset_directory", type=str, help="Path to the directory containing the samples")

    args = parser.parse_args()

    remove_masks(args.dataset_directory)
