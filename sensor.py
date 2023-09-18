from carla import BlueprintLibrary


class Sensors:
    def __init__(
        self,
        blueprint_library: BlueprintLibrary,
        transform,
        ego_vehicle,
        size=(480, 360),
    ) -> None:
        self.blueprint_library = blueprint_library
        self.transform = transform
        self.ego_vehicle = ego_vehicle
        self.size = size
        self.setRGB()
        self.setSeg()
        self.setDepth()

    def setRGB(self):
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.size[0]}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.size[1]}")
        self.rgb_cam.set_attribute("fov", "110")

    def setSeg(self):
        self.seg_cam = self.blueprint_library.find(
            "sensor.camera.semantic_segmentation"
        )
        self.seg_cam.set_attribute("image_size_x", f"{self.size[0]}")
        self.seg_cam.set_attribute("image_size_y", f"{self.size[1]}")
        self.seg_cam.set_attribute("fov", "110")

    def setDepth(self):
        self.depth_cam = self.blueprint_library.find("sensor.camera.depth")
        self.depth_cam.set_attribute("image_size_x", f"{self.size[0]}")
        self.depth_cam.set_attribute("image_size_y", f"{self.size[1]}")
        self.depth_cam.set_attribute("fov", "110")
