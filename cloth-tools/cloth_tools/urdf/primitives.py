"""Python functions to easily generate URDF for simple primitives.

See: 
"""
import xmltodict


def dict_to_urdf(dict_: dict) -> str:
    return xmltodict.unparse(dict_, pretty=True, indent="  ")


def cylinder_geometry_dict(length: float, radius: float) -> dict:
    geometry_dict = {"cylinder": {"@length": f"{length}", "@radius": f"{radius}"}}
    return geometry_dict


def cylinder_dict(length: float, radius: float) -> dict:
    geometry_dict = cylinder_geometry_dict(length, radius)
    cylinder_dict_ = {
        "robot": {
            "@name": "cylinder",
            "link": {
                "@name": "base_link",
                "visual": {"geometry": geometry_dict}, # "material": {"@name": "blue"}},
                "collision": {"geometry": geometry_dict},
            },
        }
    }
    return cylinder_dict_


def cylinder_urdf(length: float, radius: float) -> str:
    cylinder_dict_ = cylinder_dict(length, radius)
    return dict_to_urdf(cylinder_dict_)
