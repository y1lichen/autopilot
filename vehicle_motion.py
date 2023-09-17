import carla


class VehicleMotion:
    def __init__(self, vehicle, world, debug=False):
        self.vehicle: carla.Vehicle = vehicle
        self.world = world
        self.debug = debug
        self.control = carla.VehicleControl()

    # 滑行
    def coast(self):
        self.control.brake = 0
        self.control.throttle = 0
        self.vehicle.apply_control(self.control)
