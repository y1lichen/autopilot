import carla
import cv2
import numpy as np
import math
from utils.speed_meter import SpeedMeter

from vehicle_motion import VehicleMotion

client = carla.Client("localhost", 2000)
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = world.get_blueprint_library().filter("*mini*")
start_point = spawn_points[0]
ego_vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)

CAMERA_POS_Z = 3
CAMERA_POS_X = -5
SIZE = (640, 360)

camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(SIZE[0]))
camera_bp.set_attribute("image_size_y", str(SIZE[1]))

camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z, x=CAMERA_POS_X))
# this creates the camera in the sim
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)


def camera_callback(image, data_dict):
    data_dict["image"] = np.reshape(
        np.copy(image.raw_data), (image.height, image.width, 4)
    )


image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

camera_data = {"image": np.zeros((image_h, image_w, 4))}
# this actually opens a live stream from the camera
camera.listen(lambda image: camera_callback(image, camera_data))

PREFERRED_SPEED = 30  # what it says
SPEED_THRESHOLD = 2  # defines when we get close to desired speed so we drop the

# adding params to display text to image
font = cv2.FONT_HERSHEY_SIMPLEX
# org - defining lines to display telemetry values on the screen
org = (30, 30)  # this line will be used to show current speed
org2 = (30, 50)  # this line will be used for future steering angle
org3 = (30, 70)  # and another line for future telemetry outputs
org4 = (30, 90)  # and another line for future telemetry outputs
org3 = (30, 110)  # and another line for future telemetry outputs
fontScale = 0.5
# white color
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 1


#
cv2.namedWindow("RGB Camera", cv2.WINDOW_AUTOSIZE)
cv2.imshow("RGB Camera", camera_data["image"])

# main loop
quit = False

vehicle_motion = VehicleMotion(ego_vehicle, debug=True)
while True:
    # Carla Tick
    world.tick()
    if cv2.waitKey(1) == ord("q"):
        quit = True
        break
    image = camera_data["image"]

    # to get speed we need to use 'get velocity' function
    v = ego_vehicle.get_velocity()
    # if velocity is a vector in 3d
    # then speed is like hypothenuse in a right triangle
    # and 3.6 is a conversion factor from meters per second to kmh
    # e.g. kmh is 1000 meters and one hour is 60 min with 60 sec = 3600 sec
    speed = SpeedMeter.getKMHByVelocity(v)
    # now we add the speed to the window showing a camera mounted on the car
    image = cv2.putText(
        image,
        "Speed: " + str(int(speed)) + " kmh",
        org2,
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    # this is where we used the function above to determine accelerator input
    # from current speed
    vehicle_motion.cruise_control(speed, preferred_speed=80)
    cv2.imshow("RGB Camera", image)

# clean up
cv2.destroyAllWindows()
camera.stop()
for actor in world.get_actors().filter("*vehicle*"):
    actor.destroy()
for sensor in world.get_actors().filter("*sensor*"):
    sensor.destroy()
