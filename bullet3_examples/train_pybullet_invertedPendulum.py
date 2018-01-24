import numpy as np
import gym
import pybullet_envs
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv

from baselines import ddpg

experiment= 'InvertedPendulumBulletEnv-v0' #specify environments here
# 

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
	
    env = gym.make(experiment)
    env.render(mode="human")


    act = ddpg.run(env_id=experiment, 
    	     seed=123, 
    	     noise_type=['ou','ou'], 
    	     layer_norm=False, 
    	     evaluation=True)

    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
