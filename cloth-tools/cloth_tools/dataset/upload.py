import os

import requests


def upload_grasp(grasp_json_path, team_name, sample_id, server_url):
    """Uploads a grasp.json file along with team name and sample ID."""

    upload_url = server_url + "/upload_grasp"

    with open(grasp_json_path, "rb") as f:
        files = {"file": (os.path.basename(grasp_json_path), f)}
        data = {"team_name": team_name, "sample_id": sample_id}

        response = requests.post(upload_url, files=files, data=data)

    if response.status_code == 201:
        print(f"Grasp uploaded successfully: {response.json()}")
    else:
        print(f"Upload failed: {response.status_code}, {response.text}")
