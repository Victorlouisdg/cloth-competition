import os

import requests


def upload_grasp(grasp_json_path: str, team_name: str, sample_id: str, server_url: str) -> None:
    """Uploads a grasp.json file to the Cloth Competition server.

    Args:
        grasp_json_path: Path to the grasp.json file.
        team_name: Your team's name, used to organize submissions.
        sample_id: The ID of the sample that the grasp was generated for, for example: 2024-04-22_09-26-45-537535
        server_url: URL of the server to upload to.

    """
    upload_url = server_url + "/upload_grasp"

    with open(grasp_json_path, "rb") as f:
        files = {"file": (os.path.basename(grasp_json_path), f)}
        data = {"team_name": team_name, "sample_id": sample_id}

        response = requests.post(upload_url, files=files, data=data)

    if response.status_code == 201:
        print(f"Grasp uploaded successfully: {response.json()}")
    else:
        print(f"Upload failed: {response.status_code}, {response.text}")
