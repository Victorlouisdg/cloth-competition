"""A few bookkeeping functions for creating dataset files and directories.

In general we number files and directories with a suffix that is left-padded with zeros, e.g:
    dataset_0000
        - sample_000000
            sample_000000.png
            sample_000000.ply
        - sample_000001
"""
import os


def find_highest_suffix(dir: str, name: str, extension: str | None = None) -> int:
    """Find the highest suffix of files or directories in a directory with a given name.
    Only files with exactly the same extension are considered.

    For example if the directory contains files with the names:
        sample_000000.png
        sample_000001.png
        sample_000002.png
        sample_000003.jpg
        sample_000003/
    then the highest suffix is 2.

    Args:
        dir: The directory where the files or directories are considered.
        name: The basename of the files or directories.
        extension: The extension of the files.

    Returns:
        The highest suffix found.
    """
    names = os.listdir(dir)

    # Filter by extension
    extension = "" if extension is None else extension
    names_with_same_extension = [name for name in names if os.path.splitext(name)[1] == ""]

    # Remove the extensions
    names_without_extension = [os.path.splitext(name)[0] for name in names_with_same_extension]

    # Try to split the suffixes and convert them to integers
    suffixes_int = []
    for name in names_without_extension:
        try:
            suffix = name.split("_")[-1]
            suffix = suffix[:-1].lstrip("0") + suffix[-1]  # Remove leading zeros (but keep at least one)
            suffix_int = int(suffix)
            suffixes_int.append(suffix_int)
        except ValueError:
            continue

    if len(suffixes_int) == 0:
        return -1  # No suffixes found, return -1 so that adding 1 gives 0

    return max(suffixes_int)


def ensure_dataset_dir(dataset_dir: str | None = None) -> str:
    """Creates the dataset directory if it does not exist.
    If the dataset_dir is None the name is set to "dataset_000X".
    Where X is the first available suffix

    Args:
        dataset_dir: The desired path to dataset directory.

    Returns:
        The dataset directory.
    """
    if dataset_dir is None:
        dataset_dir_name = "dataset"
        dataset_suffix_int = find_highest_suffix(".", dataset_dir_name) + 1
        dataset_dir = f"{dataset_dir_name}_{dataset_suffix_int:04d}"

    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir
