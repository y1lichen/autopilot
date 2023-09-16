import carla
import pygame


class ManualControl:
    def __init__(self, vehicle, world, debug):
        self.vehicle = vehicle
        self.world = world
        self.debug = debug
        self.control = carla.VehicleControl()

        self.play()

    def play(self):
        pygame.init()
        size = (360, 360)
        pygame.display.set_caption("CARLA Manual Control")
        pygame.display.set_mode(size)
        clock = pygame.time.Clock()
        done = False
        while not done:
            keys = pygame.key.get_pressed()

            self.goFroward(keys)
            self.goBackward(keys)
            self.turn(keys)
            self.hand_brake(keys)

            self.world.tick()
            # Update the display and check for the quit event
            pygame.display.flip()
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # Sleep to ensure consistent loop timing
            clock.tick(60)

    def goFroward(self, keys):
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.control.throttle = min(self.control.throttle + 0.05, 1.0)
        else:
            self.control.throttle = 0.0
        self.vehicle.apply_control(self.control)

    def goBackward(self, keys):
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.control.brake = min(self.control.brake + 0.2, 1.0)
        else:
            self.control.brake = 0.0

    def turn(self, keys):
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.control.steer = max(self.control.steer - 0.05, -1.0)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.control.steer = min(self.control.steer + 0.05, 1.0)
        else:
            self.control.steer = 0.0
        self.vehicle.apply_control(self.control)

    def hand_brake(self, keys):
        self.control.hand_brake = keys[pygame.K_SPACE]
        self.vehicle.apply_control(self.control)
