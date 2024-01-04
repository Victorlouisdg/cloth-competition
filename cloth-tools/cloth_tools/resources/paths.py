import os

import cloth_tools


def get_resource_path(name: str):
    filepath = os.path.realpath(__file__)
    resource_path = os.path.join(os.path.dirname(filepath), name)
    return resource_path


table = get_resource_path("table.urdf")

if __name__ == "__main__":
    print(cloth_tools.resources.table)
