import os
import urllib
import zipfile

import requests
from tqdm import tqdm


def download_and_extract_dataset(data_dir: str, dataset_zip_url: str) -> str:
    """Download and extract a Cloth Competition Dataset zip file from a URL.

    Args:
        data_dir: The parent directory to save the extracted dataset to.
        dataset_zip_url: The URL of the dataset zip file.

    Returns:
        The path to the extracted dataset directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    with requests.get(dataset_zip_url, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB at a time

        zip_name = os.path.basename(urllib.parse.urlparse(dataset_zip_url).path)
        zip_path = os.path.join(data_dir, zip_name)
        dataset_dir = os.path.join(data_dir, zip_name.split(".")[0])

        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)

    return dataset_dir
