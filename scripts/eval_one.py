#!/usr/bin/env python
"""One Cheng-style eval: 256 envs, 20 s episodes, 1500 steps.

Based on extreme-parkour legged_gym/legged_gym/scripts/evaluate.py.
Re-applies eval terrain after helpers.py camera-headless overwrites.
"""
import argparse
import json
import os
import statistics
import sys
from collections import deque

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(ROOT, "legged_gym"))
os.chdir(os.path.join(ROOT, "legged_gym", "legged_gym", "scripts"))

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, update_cfg_from_args, set_seed

EMPTY_TERRAIN = {
    "smooth slope": 0.0,
    "rough slope up": 0.0,
    "rough slope down": 0.0,
    "rough stairs up": 0.0,
    "rough stairs down": 0.0,
    "discrete": 0.0,
    "stepping stones": 0.0,
    "gaps": 0.0,
    "smooth flat": 0.0,
    "pit": 0.0,
    "wall": 0.0,
    "platform": 0.0,
    "large stairs up": 0.0,
    "large stairs down": 0.0,
    "parkour": 0.0,
    "parkour_hurdle": 0.0,
    "parkour_flat": 0.0,
    "parkour_step": 0.0,
    "parkour_gap": 0.0,
    "demo": 0.0,
}

TERRAINS = {
    "mixed": {"parkour": 0.25, "parkour_hurdle": 0.25, "parkour_step": 0.25, "parkour_gap": 0.25},
    "hurdle": {"parkour_hurdle": 1.0},
    "step": {"parkour_step": 1.0},
    "gap": {"parkour_gap": 1.0},
    "ramps": {"parkour": 1.0},
}


def mean_std(xs):
    xs = list(xs)
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def make_eval_env(args, env_cfg, terrain_weights, num_envs):
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    d = dict(EMPTY_TERRAIN)
    d.update(terrain_weights)
    env_cfg.terrain.terrain_dict = d
    env_cfg.terrain.terrain_proportions = list(d.values())
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    set_seed(env_cfg.seed)
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    task_class = task_registry.get_task_class(args.task)
    env = task_class(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    return env, env_cfg


def main():
    pre = argparse.ArgumentParser()
    pre.add_argument("--eval_exptid", required=True)
    pre.add_argument("--eval_checkpoint", type=int, required=True)
    pre.add_argument("--eval_terrain", required=True, choices=list(TERRAINS))
    pre.add_argument("--eval_camera", action="store_true")
    pre.add_argument("--eval_num_envs", type=int, default=256)
    pre.add_argument("--eval_tag", default="")
    pre.add_argument(
        "--eval_fusion_mode",
        default="concat",
        choices=["concat", "film_prop", "hyper_last", "film_vis", "film_cat"],
    )
    pre.add_argument("--eval_student_visual", default="depth", choices=["depth", "scandots"])
    pre.add_argument(
        "--eval_jsonl",
        default=os.path.join(ROOT, "results", "cheng_eval.jsonl"),
        help="Append one JSON record per eval.",
    )
    known, rest = pre.parse_known_args()
    argv = [
        "evaluate.py",
        "--exptid", known.eval_exptid,
        "--checkpoint", str(known.eval_checkpoint),
        "--delay",
        "--device", "cuda:0",
        "--no_wandb",
        "--headless",
        "--num_envs", str(known.eval_num_envs),
        "--fusion_mode", known.eval_fusion_mode,
        "--student_visual", known.eval_student_visual,
    ]
    if known.eval_camera:
        argv.append("--use_camera")
    argv += rest
    sys.argv = argv

    if known.eval_camera:
        _orig = LeggedRobot.create_sim

        def _keep_gfx(self):
            self.graphics_device_id = self.sim_device_id
            return _orig(self)

        LeggedRobot.create_sim = _keep_gfx

    args = get_args()
    log_root = os.path.join(ROOT, "legged_gym", "logs", args.proj_name, args.exptid)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env, env_cfg = make_eval_env(args, env_cfg, TERRAINS[known.eval_terrain], known.eval_num_envs)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _log = task_registry.make_alg_runner(
        log_root=log_root,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
        init_wandb=False,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    total_steps = 1000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)

    for _i in tqdm(range(1500), desc="{} {}".format(known.eval_tag, known.eval_terrain)):
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, : env.cfg.env.n_proprio]
                obs_student[:, 6:8] = 0
                with torch.no_grad():
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5 * yaw
        else:
            depth_latent = None
        if known.eval_student_visual == "scandots":
            with torch.no_grad():
                actor_z = ppo_runner.alg.actor_critic.actor.infer_scandots_latent(obs.detach())
        else:
            actor_z = depth_latent
        if hasattr(ppo_runner.alg, "depth_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=actor_z)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=actor_z)
        cur_goal_idx = env.cur_goal_idx.clone()
        obs, _, rews, dones, infos = env.step(actions.detach())
        edge_violation_buffer.extend(env.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist())
        cur_reward_sum += rews
        cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        num_waypoints_buffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0

    wp = np.array(num_waypoints_buffer, dtype=float) / 7.0
    rew_m, rew_s = mean_std(rewbuffer)
    len_m, len_s = mean_std(lenbuffer)
    mxd_m = float(np.mean(wp)) if wp.size else float("nan")
    mxd_s = float(np.std(wp)) if wp.size else float("nan")
    mev_m = float(np.mean(edge_violation_buffer)) if len(edge_violation_buffer) else float("nan")
    mev_s = float(np.std(edge_violation_buffer)) if len(edge_violation_buffer) else float("nan")
    rec = {
        "tag": known.eval_tag,
        "exptid": known.eval_exptid,
        "checkpoint": known.eval_checkpoint,
        "terrain": known.eval_terrain,
        "use_camera": bool(known.eval_camera),
        "fusion_mode": known.eval_fusion_mode,
        "student_visual": known.eval_student_visual,
        "num_envs": int(env.num_envs),
        "n_episodes": len(rewbuffer),
        "rew_mean": rew_m,
        "rew_std": rew_s,
        "len_mean": len_m,
        "len_std": len_s,
        "mxd_mean": mxd_m,
        "mxd_std": mxd_s,
        "mev_mean": mev_m,
        "mev_std": mev_s,
    }
    os.makedirs(os.path.dirname(os.path.abspath(known.eval_jsonl)) or ".", exist_ok=True)
    with open(known.eval_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("EVAL_JSON", json.dumps(rec))
    print("Mean reward: {:.2f}$\\pm${:.2f}".format(rew_m, rew_s))
    print("Mean episode length: {:.2f}$\\pm${:.2f}".format(len_m, len_s))
    print("Mean number of waypoints: {:.2f}$\\pm${:.2f}".format(mxd_m, mxd_s))
    print("Mean edge violation: {:.2f}$\\pm${:.2f}".format(mev_m, mev_s))
    print("EVAL_OK", known.eval_tag, known.eval_terrain, "envs", env.num_envs)


if __name__ == "__main__":
    main()
