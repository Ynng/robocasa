import random
import time
from typing import Dict
import unittest

import numpy as np
import robocasa
import robosuite
from robosuite import load_controller_config
from termcolor import colored


MOVEMENT_THRESHOLD = 0.1
DEFAULT_SEED = 3

# TODO: get the test to work with all environments
# TODO: reduce the amount of time it takes to run
# Currently this test takes 10*100*5 seconds = 1.4 hours to run per Environment


class TestMujocoPhysicsStability(unittest.TestCase):
    verbose = False
    test_envs = [
        # "PnPCounterToCab",
        # "PnPCounterToSink",
        # "PnPMicrowaveToCounter",
        # "PnPStoveToCounter",
        # "OpenSingleDoor",
        # "CloseDrawer",
        # "TurnOnMicrowave",
        # "TurnOnSinkFaucet",
        # "TurnOnStove",
        # "ArrangeVegetables",
        # "MicrowaveThawing",
        # "RestockPantry",
        "PreSoakPan",
        # "PrepareCoffee",
    ]

    def test_mujoco_physics_stability(self):
        """
        Tests that the physics of the mujoco simulator are stable.
        This test does the following:
        - Runs every task with every layout and a random style 100 times
        - For each run, simulates 60 steps
        - Checks that the objects do not move more than MOVEMENT_THRESHOLD
        This ensures that the physics simulation remains consistent and stable
        across different configurations and multiple iterations.
        """

        def get_object_positions(env) -> Dict[str, np.ndarray]:
            return {
                obj_name: np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
                for obj_name, obj in env.objects.items()
            }

        def check_object_movement(
            env, original_positions: Dict[str, np.ndarray]
        ) -> None:
            current_positions = get_object_positions(env)
            for obj_name, current_pos in current_positions.items():
                if obj_name in original_positions:
                    distance = np.linalg.norm(
                        current_pos - original_positions[obj_name]
                    )
                    if distance > MOVEMENT_THRESHOLD:
                        current_time = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime()
                        )
                        error_message = (
                            f"{obj_name} moved unexpectedly during reset. "
                            f"Time: {current_time}\n"
                            f"Current position: {current_pos}\n"
                            f"Original position: {original_positions[obj_name]}"
                        )
                        raise AssertionError(error_message)

        # envs = sorted(robocasa.ALL_KITCHEN_ENVIRONMENTS)

        for _, env in enumerate(self.test_envs):
            print(colored(f"Testing {env} environment...", "green"))

            config = {
                "env_name": env,
                "robots": "PandaMobile",
                "controller_configs": load_controller_config(
                    default_controller="OSC_POSE"
                ),
                "has_renderer": True,
                "has_offscreen_renderer": False,
                "render_camera": "robot0_frontview",
                "ignore_done": True,
                "renderer": "mjviewer",
                "use_camera_obs": False,
                "seed": DEFAULT_SEED,
                "randomize_cameras": False,
                "obj_registries": ("objaverse", "aigen"),
            }

            # Try every layout & style configuration
            for layout in range(10):
                config["layout_ids"] = layout
                config["style_ids"] = random.randint(0, 11)
                env: robocasa.Kitchen = robosuite.make(**config)
                for _ in range(100):  # Try 100 times
                    # env.reset() resets the sim and randomizes object placements
                    env.reset()
                    initial_pos = get_object_positions(env)
                    if self.verbose:
                        print(
                            f"initial_pos for layout {layout} style {config['style_ids']}: {initial_pos}"
                        )

                    for _ in range(60):  # Step 60 times
                        start_time = time.time()

                        env.step([0] * env.action_dim)
                        check_object_movement(env, initial_pos)

                        elapsed_time = time.time() - start_time
                        sleep_time = max(1 / 60 - elapsed_time, 0)
                        time.sleep(sleep_time)

                    env.close()
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    TestMujocoPhysicsStability.verbose = args.verbose
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
