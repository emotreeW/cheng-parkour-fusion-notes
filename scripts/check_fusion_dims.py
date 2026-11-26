#!/usr/bin/env python
"""Forward-shape check for fusion_mode. No Isaac Gym."""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "rsl_rl"))

from rsl_rl.modules.actor_critic import Actor, copy_compatible_actor, get_activation, FUSION_MODES
from rsl_rl.modules.depth_backbone import RecurrentDepthBackbone

act = get_activation("elu")
kw = dict(
    num_prop=53,
    num_scan=132,
    num_actions=12,
    scan_encoder_dims=[128, 64, 32],
    actor_hidden_dims=[512, 256, 128],
    priv_encoder_dims=[64, 20],
    num_priv_latent=29,
    num_priv_explicit=9,
    num_hist=10,
    activation=act,
)
obs_dim = 53 + 132 + 9 + 29 + 10 * 53
B = 4
obs = torch.randn(B, obs_dim)
z = torch.randn(B, 32)
teacher = Actor(fusion_mode="concat", **kw)
assert teacher.actor_backbone[0].in_features == 114, teacher.actor_backbone[0].in_features

for mode in FUSION_MODES:
    stu = Actor(fusion_mode=mode, **kw)
    in_f = stu.actor_backbone[0].in_features
    copy_compatible_actor(teacher, stu)
    if mode in ("film_prop", "hyper_last"):
        assert in_f == 82, (mode, in_f)
    else:
        assert in_f == 114, (mode, in_f)
    out = stu(obs, hist_encoding=True, scandots_latent=z)
    assert out.shape == (B, 12), (mode, tuple(out.shape))
    assert torch.isfinite(out).all(), mode
    print("actor_ok", mode, "in_features", in_f, "out", tuple(out.shape))


class DummyCfg:
    class env:
        n_proprio = 53


enc = RecurrentDepthBackbone(torch.nn.Identity(), DummyCfg, fusion_mode="film_vis")
depth = torch.randn(B, 32)
prop = torch.randn(B, 53)
y = enc(depth, prop)
assert y.shape == (B, 34), y.shape
assert torch.isfinite(y).all()
enc_cat = RecurrentDepthBackbone(torch.nn.Identity(), DummyCfg, fusion_mode="concat")
y2 = enc_cat(depth, prop)
assert y2.shape == (B, 34)
print("encoder_ok film_vis", tuple(y.shape), "concat", tuple(y2.shape))
print("DIM_OK")
