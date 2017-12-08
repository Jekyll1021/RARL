"""
Training script for single agent framework
"""

import argparse
import logging
import os.path as osp

import MlpPolicy
import gym
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from reimplementation import PPO


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)


def train(env_id, num_iters, seed):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    test_env = gym.make(env_id)

    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    test_env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    rew = PPO.learn(env, test_env, policy_fn,
                    max_iters=num_iters,
                    timesteps_per_batch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                    gamma=0.99, lam=0.95, schedule='constant',
                    )
    env.close()
    return rew


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    model = train(args.env, num_iters=1000, seed=args.seed)
    print(model)
