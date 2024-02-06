import airo_models


def create_static_robotiq_2f_85_urdf() -> str:
    robotiq_urdf_path = airo_models.get_urdf_path("robotiq_2f_85")
    robotiq_urdf = airo_models.urdf.read_urdf(robotiq_urdf_path)

    # Make the robotiq gripper static
    airo_models.urdf.replace_value(robotiq_urdf, "@type", "revolute", "fixed")
    airo_models.urdf.delete_key(robotiq_urdf, "mimic")
    airo_models.urdf.delete_key(robotiq_urdf, "transmission")

    # Write it to a temporary file to read later with Drake's AddModels
    robotiq_static_urdf_path = airo_models.urdf.write_urdf_to_tempfile(
        robotiq_urdf, robotiq_urdf_path, prefix="robotiq_2f_85_static_"
    )
    return robotiq_static_urdf_path
