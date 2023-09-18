import carla
import math


class SpeedMeter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def getMPHByVelocity(velocity: carla.Vector3D):
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    @staticmethod
    def getKMHByVelocity(velocity: carla.Vector3D):
        return 3.6 * SpeedMeter.getMPHByVelocity(velocity)
