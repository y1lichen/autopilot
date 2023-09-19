import cv2
import carla
import random
import numpy as np

from utils.speed_meter import SpeedMeter
from vehicle_motion import VehicleMotion


def main():
    # Create client and send request to the carla server.
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    # Retrieve the world that is currently running
    world = client.get_world()
    # Change Town map
    # world = client.load_world('Town01')

    # set sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()
    # randomly choose one as the start point
    spawn_point = random.choice(spawn_points)

    ego_vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2017")
    # ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    # ego_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

    # let CarlaViz know this vehicle is the ego vehicle.
    ego_vehicle_bp.set_attribute("role_name", "ego")
    # spawn the vehicle
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

    rgb_cam_bp = None
    rgb_cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    rgb_cam_bp.set_attribute("image_size_x", str(640))
    rgb_cam_bp.set_attribute("image_size_y", str(480))
    rgb_cam_bp.set_attribute("fov", str(52))
    rgb_cam_location = carla.Location(2, 0, 1)
    rgb_cam_rotation = carla.Rotation(0, 0, 0)
    rgb_cam_transform = carla.Transform(rgb_cam_location, rgb_cam_rotation)
    rgb_cam = world.spawn_actor(
        rgb_cam_bp,
        rgb_cam_transform,
        attach_to=ego_vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    def camera_callback(image, data_dict):
        data_dict["image"] = np.reshape(
            np.copy(image.raw_data), (image.height, image.width, 4)
        )

    rgb_cam.listen(lambda image: camera_callback(image, camera_data))

    image_w = rgb_cam_bp.get_attribute("image_size_x").as_int()
    image_h = rgb_cam_bp.get_attribute("image_size_y").as_int()

    camera_data = {"image": np.zeros((image_h, image_w, 4))}

    cv2.namedWindow("RGB Camera", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("RGB Camera", camera_data["image"])

    vehicle_motion = VehicleMotion(ego_vehicle, debug=True)
    while True:
        world.tick()
        if cv2.waitKey(1) == ord("q"):
            break
        ego_location = ego_vehicle.get_location()
        ego_velocity = ego_vehicle.get_velocity()
        cur_speed = SpeedMeter.getKMHByVelocity(ego_velocity)

        image = camera_data["image"]
        image = cv2.putText(
            image,
            "Speed: " + str(int(cur_speed)) + " kmh",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # white
            1,
            cv2.LINE_AA,
        )
        vehicle_motion.cruise_control(current_speed=cur_speed, preferred_speed=60)
        cv2.imshow("RGB Camera", image)
    cv2.destroyAllWindows()
    rgb_cam.stop()
    for actor in world.get_actors().filter("*vehicle*"):
        actor.destroy()
    for sensor in world.get_actors().filter("*sensor*"):
        sensor.destroy()
    print("destroying actors done")


if __name__ == "__main__":
    main()
