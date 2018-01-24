import argparse
import time
import os
import logging
import pybullet_envs
from baselines.ddpg_python2.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.ddpg_python2.config import exp_config
import baselines.ddpg_python2.training as training
from baselines.ddpg_python2.models import Actor, Critic
from baselines.ddpg_python2.memory import Memory
from baselines.ddpg_python2.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI


def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()


    # Create envs.
    env = gym.make(env_id)
    # env.render(mode="human")

    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        print ('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':

    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters

    if exp_config['num_timesteps'] is not None:
        assert(exp_config['num_timesteps'] == exp_config['nb_epochs'] * exp_config['nb_epoch_cycles'] * exp_config['nb_rollout_steps'])

    del exp_config['num_timesteps']

    # Run actual script.
    run(**exp_config)
