    
ddpg_experiment_config = {'env_id':'InvertedPendulumBulletEnv-v0',
                          'render_eval':False,
                          'layer_norm':True,
                          'render':True,
                          'normalize_returns':False,
                          'normalize_observations':True,
                          'seed':0,
                          'critic_l2_reg':1e-2,
                          'batch_size':64,
                          'actor_lr':1e-4, # per MPI worker
                          'critic_lr':1e-3,
                          'popart':False,
                          'gamma':0.99,
                          'reward_scale':1.,
                          'clip_norm':None,
                          'nb_epochs':500, # with default settings, perform 1M steps total
                          'nb_epoch_cycles':20,
                          'nb_train_steps':50, # per epoch cycle and MPI worker
                          'nb_eval_steps':100, # per epoch cycle and MPI worker
                          'nb_rollout_steps':100, # per epoch cycle and MPI worker
                          'noise_type':'adaptive-param_0.2', # choices are adaptive-param_xx, ou_xx, normal_xx, none
                          'num_timesteps':None,
                          'evaluation':False

                         }



exp_config = ddpg_experiment_config