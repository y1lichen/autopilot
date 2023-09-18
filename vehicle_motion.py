import carla


class VehicleMotion:
    def __init__(self, vehicle, debug=False):
        self.vehicle: carla.Vehicle = vehicle
        self.steering_angle = 0
        self.debug = debug
        self.control = carla.VehicleControl()

    # 滑行
    def coast(self):
        self.releaseBrakeAndThrottle()
        self.vehicle.apply_control(self.control)

    def releaseBrakeAndThrottle(self):
        self.control.brake = 0
        self.control.throttle = 0

    # 利用調整油門開度達到定速
    def cruise_control(self, current_speed, preferred_speed, threshold=2):
        self.releaseBrakeAndThrottle()
        if current_speed < threshold:
            self.control.throttle = 0.18
        elif current_speed >= preferred_speed:
            self.control.throttle = 0
        elif current_speed < preferred_speed:
            self.control.throttle = max(
                preferred_speed - current_speed / abs(preferred_speed - threshold), 0.2
            )
        if self.debug:
            print(f"速度：{current_speed}, 油門開度：{self.control.throttle}")
        self.vehicle.apply_control(self.control)
