import carla
import random

from manual_control import ManualControl


class Main:
    def __init__(self, manual_control=False, debug=False):
        self.manual_control = manual_control
        self.debug = debug

        self.client = carla.Client("localhost", 2000)
        self.world = self.client.get_world()

        # world = client.load_world("Town06")
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=10.0,
            sun_azimuth_angle=70.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
        )
        self.world.set_weather(weather)

        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        vehicle_bp = bp_lib.find("vehicle.audi.etron")
        self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[79])

        self.spectator = self.world.get_spectator()
        self.transform = carla.Transform(
            self.ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
            self.ego_vehicle.get_transform().rotation,
        )
        self.spectator.set_transform(self.transform)

        for _ in range(70):
            vehicle_bp = random.choice(bp_lib.filter("vehicle"))
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        for v in self.world.get_actors().filter("*vehicle*"):
            v.set_autopilot(True)
        self.ego_vehicle.set_autopilot(False)

    def play(self):
        if self.manual_control:
            ManualControl(self.ego_vehicle, self.world, self.debug)


if __name__ == "__main__":
    main = Main(manual_control=True, debug=True)
    main.play()
