import argparse
import json
import os
import random
import time

import h5py
import robosuite

import numpy as np
from robocasa.scripts.download_datasets import download_datasets
from robocasa.scripts.playback_dataset import (
    get_env_metadata_from_dataset,
    playback_dataset,
    playback_trajectory_with_env,
    playback_trajectory_with_obs,
    reset_to,
)
from termcolor import colored
from robocasa.utils.dataset_registry import get_ds_path


TASK = "PreSoakPan"

if __name__ == "__main__":
    video_num = 0
    dataset = get_ds_path(TASK, ds_type="human_raw")
    if os.path.exists(dataset) is False:  # type: ignore
        # download dataset files
        print(colored("Unable to find dataset locally. Downloading...", color="yellow"))
        download_datasets(tasks=[TASK], ds_types=["human_raw"])

    parser = argparse.Namespace()
    parser.dataset = dataset
    parser.render = True
    parser.video_path = False
    parser.use_actions = False
    parser.use_abs_actions = False
    parser.render_image_names = ["robot0_agentview_center"]
    parser.use_obs = False
    parser.n = None
    parser.filter_key = None
    parser.video_skip = 5
    parser.first = False
    parser.verbose = True
    parser.extend_states = True
    args = parser

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.use_abs_actions:
        env_meta["env_kwargs"]["controller_configs"][
            "control_delta"
        ] = False  # absolute action space

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False

    if args.verbose:
        print(
            colored(
                "Initializing environment for {}...".format(env_kwargs["env_name"]),
                "yellow",
            )
        )

    env = robosuite.make(**env_kwargs)
    print("Env initialized", type(env))

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if "data" in f.keys():
        demos = list(f["data"].keys())  # type: ignore

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        video_count = 0

        if args.verbose:
            ep_meta = json.loads(initial_state["ep_meta"])
            lang = ep_meta.get("lang", None)
            if lang is not None:
                print(colored(f"Instruction: {lang}", "green"))
            print(colored("Spawning environment...", "yellow"))
        
        reset_to(env, initial_state)

        traj_len = states.shape[0]

        # grab all objects initial position
        obejcts_last_pos = {}
        for obj_name, obj in env.objects.items():
            obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
            obejcts_last_pos[obj_name] = obj_pos
            print(f"{obj_name} initial position: {obj_pos}")
        
        # reset each environment 100 times
        for i in range(100):
            # step 100 times to see if physics bugs occur
            for j in range(100):
                start = time.time()

                env.step([0] * env.action_dim)
                for obj_name, obj in env.objects.items():
                    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
                    if obj_name in obejcts_last_pos:
                        obj_dist = np.linalg.norm(obj_pos - obejcts_last_pos[obj_name])

                        if obj_dist > 0.01:
                            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f"{obj_name} is moving when it shouldn't be during reset {i} step {j} at time {current_time}. Current position: {obj_pos}")
                    obejcts_last_pos[obj_name] = obj_pos

                # on-screen render
                if env.viewer is None:
                    env.initialize_renderer()

                # so that mujoco viewer renders
                env.viewer.update()

                max_fr = 60
                elapsed = time.time() - start
                diff = 1 / max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)

        print("Resetting environment")
        env.reset()
        reset_to(env, initial_state)

    f.close()

    if env is not None:
        env.close()
