import time
import robosuite
from robosuite import load_controller_config
import numpy as np

TASK = "PreSoakPan"
MAX_FR = 60

if __name__ == "__main__":
    config = {
        "env_name": TASK,
        "robots": "PandaMobile",
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "layout_ids": None,
        "style_ids": None,
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "render_camera": "robot0_frontview",
        "ignore_done": True,
        "renderer": "mjviewer",
        "use_camera_obs": False,
    }

    # reset each environment 100 times
    env = robosuite.make(**config)
    for i in range(100):
        for j in range(10):
            # grab all objects initial position
            obejcts_last_pos = {}
            for obj_name, obj in env.objects.items():
                obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
                obejcts_last_pos[obj_name] = obj_pos
                print(f"{obj_name} initial position: {obj_pos}")

            for k in range(60):
                start = time.time()

                env.step([0] * env.action_dim)
                for obj_name, obj in env.objects.items():
                    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
                    if obj_name in obejcts_last_pos:
                        obj_dist = np.linalg.norm(obj_pos - obejcts_last_pos[obj_name])

                        if obj_dist > 0.01:
                            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f"{obj_name} is moving when it shouldn't be during reset {j} step {k} at time {current_time}. Current position: {obj_pos}")
                    obejcts_last_pos[obj_name] = obj_pos

                # on-screen render
                if env.viewer is None:
                    env.initialize_renderer()

                # so that mujoco viewer renders
                env.viewer.update()

                elapsed = time.time() - start
                diff = 1 / MAX_FR - elapsed
                if diff > 0:
                    time.sleep(diff)

            env.close()
            env.reset()
