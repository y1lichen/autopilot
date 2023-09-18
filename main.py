import carla
import random
import time
import numpy as np
import cv2

from sensor import Sensors
from vehicle_motion import VehicleMotion


class CarlaEnv:
    actors_list = []
    front_cam = None
    front_depth_cam = None
    front_seg_cam = None
    size = (480, 360)

    def __init__(self, debug=False) -> None:
        self.debug = debug

        self.world: carla.World = client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter("vehicle.tesla.model3")[0]

    def reset(self):
        self.actors_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(self.model3, self.transform)
        self.ego_vehicle_control: VehicleMotion = VehicleMotion(
            self.ego_vehicle, self.world, self.debug
        )
        self.actors_list.append(self.ego_vehicle)
        self.spawn_camera()
        time.sleep(5)

        while self.front_camera is None:
            time.sleep(0.01)
        while self.front_depth_camera is None:
            time.sleep(0.01)
        while self.front_seg_camera is None:
            time.sleep(0.01)

    def spawn_camera(self):
        sensors = Sensors(
            self.blueprint_library, self.transform, self.ego_vehicle, self.size
        )
        self.rgb_sensor = self.world.spawn_actor(
            sensors.rgb_cam, self.transform, attach_to=self.ego_vehicle
        )
        self.actors_list.append(self.rgb_sensor)
        self.rgb_sensor.listen(lambda date: self.process_img(date))

        self.seg_sensor = self.world.spawn_actor(
            sensors.seg_cam, self.transform, attach_to=self.ego_vehicle
        )
        self.actors_list.append(self.seg_sensor)
        self.seg_sensor.listen(lambda date: self.seg_filter(date))

        self.depth_sensor = self.world.spawn_actor(
            sensors.depth_cam, self.transform, attach_to=self.ego_vehicle
        )
        self.actors_list.append(self.depth_sensor)
        self.depth_sensor.listen(lambda data: self.process_depth_img(data))
        self.ego_vehicle_control.coast()

    def destroy(self):
        for act in self.actors_list:
            act.destroy()

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.size[1], self.size[0], 4))
        i3 = i2[:, :, :3]
        if self.debug:
            cv2.imshow("", i3)
        self.front_camera = i3

    def seg_filter(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.size[1], self.size[0], 4))
        i3 = i2[:, :, :3]
        i3[:, :, 0] = 255 * (i3[:, :, 2] == 6)
        i3[:, :, 1] = 255 * (i3[:, :, 2] == 7)
        i3[:, :, 2] = 255 * (i3[:, :, 2] == 10)
        self.front_seg_camera = i3
        print(self.front_seg_camera.shape)

    def process_depth_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.size[1], self.size[0], 4))
        i3 = i2[:, :, :3]
        if self.debug:
            cv2.imshow("", i3)
        normalized = (i3[:, :, 2] + i3[:, :, 1] * 256 + i3[:, :, 0] * 256 * 256) / (
            256 * 256 * 256 - 1
        )
        self.front_depth_camera = 255 * normalized


if __name__ == "__main__":
    env = CarlaEnv(debug=True)
    env.reset()
    cv2.imshow("", env.front_cam)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    env.destroy()
