import argparse
from collections import OrderedDict
import random
import time
from typing import Dict

import numpy as np
import robosuite
from robosuite import load_controller_config

import robocasa.environments.kitchen.kitchen as kitchen
MOVEMENT_THRESHOLD = 0.1

def get_object_positions(env) -> Dict[str, np.ndarray]:
    return {
        obj_name: np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
        for obj_name, obj in env.objects.items()
    }

def check_object_movement(env, last_positions: Dict[str, np.ndarray]) -> bool:
    current_positions = get_object_positions(env)
    for obj_name, current_pos in current_positions.items():
        if obj_name in last_positions:
            distance = np.linalg.norm(current_pos - last_positions[obj_name])
            if distance > MOVEMENT_THRESHOLD:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"{obj_name} moved unexpectedly during reset. Time: {current_time}")
                print(f"Current position: {current_pos}")
                return False
    return True


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task (choose among 100+ tasks)")
    args = parser.parse_args()
    
    tasks = OrderedDict(
        [
            ("PnPCounterToCab", "pick and place from counter to cabinet"),
            ("PnPCounterToSink", "pick and place from counter to sink"),
            ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PnPStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ]
    )

    if args.task is None:
        args.task = choose_option(
            tasks, "task", default="PnPCounterToCab", show_keys=True
        )
        
    config = {
        "env_name": args.task,
        "robots": "PandaMobile",
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "render_camera": "robot0_frontview",
        "ignore_done": True,
        "renderer": "mjviewer",
        "use_camera_obs": False,
    }


    for _ in range(500):
        # Try the same layout & style 100 times
        # layout is a number 0-9
        # style is a number 0-11
        layout = random.randint(0, 9)
        style = random.randint(0, 11)
        config["layout_ids"] = layout
        config["style_ids"] = style
        config["seed"] = random.randint(0, 2**32 - 1)
        env: kitchen.Kitchen = robosuite.make(**config)
        print("env type is", type(env))
        for _ in range(100):
            # env.reset() resets the sim and re-randomizes the object placements
            env.reset()
            initial_pos = get_object_positions(env)
            print("initial_pos", initial_pos)

            for _ in range(60):
                start_time = time.time()

                env.step([0] * env.action_dim)
                if not check_object_movement(env, initial_pos):
                    input("Object moved unexpectedly. Press Enter to continue...")
                    break

                elapsed_time = time.time() - start_time
                sleep_time = max(1 / 60 - elapsed_time, 0)
                time.sleep(sleep_time)

            env.close()
    env.close()

if __name__ == "__main__":
    main()
