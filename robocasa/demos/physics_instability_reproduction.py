import time
from typing import Dict

import numpy as np
import robosuite
from robosuite import load_controller_config

TASK = "PreSoakPan"
MOVEMENT_THRESHOLD = 0.1

def get_object_positions(env) -> Dict[str, np.ndarray]:
    return {
        obj_name: np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
        for obj_name, obj in env.objects.items()
    }

def check_object_movement(env, last_positions: Dict[str, np.ndarray]) -> None:
    current_positions = get_object_positions(env)
    for obj_name, current_pos in current_positions.items():
        if obj_name in last_positions:
            distance = np.linalg.norm(current_pos - last_positions[obj_name])
            if distance > MOVEMENT_THRESHOLD:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{obj_name} moved unexpectedly during reset. Time: {current_time}")
                print(f"Current position: {current_pos}")
                print(f"Pausing for 10 seconds for observation")
                time.sleep(10)

def main():
    config = {
        "env_name": TASK,
        "robots": "PandaMobile",
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "render_camera": "robot0_frontview",
        "ignore_done": True,
        "renderer": "mjviewer",
        "use_camera_obs": False,
    }

    env = robosuite.make(**config)

    for _ in range(500):
        env.reset()
        last_positions = get_object_positions(env)
        for _ in range(60):
            start_time = time.time()

            env.step([0] * env.action_dim)
            check_object_movement(env, last_positions)

            if env.viewer is None:
                env.initialize_renderer()
            env.viewer.update()

            elapsed_time = time.time() - start_time
            sleep_time = max(1 / 60 - elapsed_time, 0)
            time.sleep(sleep_time)

        env.close()
    env.close()

if __name__ == "__main__":
    main()
