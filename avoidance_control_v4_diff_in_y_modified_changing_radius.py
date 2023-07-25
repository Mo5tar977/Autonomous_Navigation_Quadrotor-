# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:31:22 2022

@author: HP
"""

from droneaviary_v2_modified_changing_radius import *
################################################################################
"""# **Update Callback**"""

class UpdateDesiredCallback(BaseCallback):
    """
    A custom callback that changes the desired and initial position of the environment every 10 episodes (5000 timesteps) .

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, training_env, verbose=0):
        super(UpdateDesiredCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.episodeCount = 0       # Count of the number of episodes
        self.train_env = train_env 

    def _on_training_start(self):
        
        self.episodeCount = 0

    def _on_rollout_start(self):
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.episodeCount = self.episodeCount + 1
        if self.episodeCount % 2 == 0:
            self.train_env.update_env()
            print(self.num_timesteps)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


################################################################################
def createTrajectory(initial, target, inc):
    
    step = np.abs((target - initial)) / inc
    step = np.max(step)
    inc = (target - initial) / step
    traj = initial
    
    for i in range(int(step)):
        initial[0] = initial[0] + inc
        traj = np.append(traj, initial, axis=0)
        
    return traj[1:]
################################################################################

path = 'results/policy_position_error_without_xyz_td3_gaussian'
pf_model = TD3.load(path + '/success_model.zip')
 
## **Train Environment**


filename = 'results/save-'+"DroneAviary"+'-'+"DDPG-paper"+'-'+"Avoidance-v2"+'-'+"RPM"+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
if not os.path.exists(filename):
  print("True")
  os.makedirs("./"+ filename)
print(filename)

train_env = DroneAviary(gui=False,
                        position_agent=pf_model)
train_env.update_env()
print("[INFO] Action space:", train_env.action_space)
print("[INFO] Observation space:", train_env.observation_space)

#check_env(train_env)

## **PPO Algorithm**
"""
onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[dict(vf=[256, 128], pi=[256, 128])]
                           )
oa_model = PPO(a2cppoMlpPolicy,
                    train_env,
                    use_sde=True,
                    learning_rate=0.0001,
                    policy_kwargs=onpolicy_kwargs,
                    batch_size=64,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )
"""
offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(qf=[400, 300], pi=[400, 300]))
oa_model = DDPG(td3ddpgMlpPolicy,
                    train_env,
                    learning_rate=0.0001,
                    learning_starts=1e4,
                    batch_size=64,
                    action_noise=NormalActionNoise(0,0.2),
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )



## **Evaluation Environment**

eval_env = DroneAviary(randomize_init=True,
                       position_agent=pf_model
                       )
eval_env.update_env()
print("[INFO] Action space:", eval_env.action_space)
print("[INFO] Observation space:", eval_env.observation_space)

"""## **Training**"""

eval_callback = EvalCallback(eval_env,
                            #callback_on_new_best=callback_on_best,
                            n_eval_episodes = 50,
                            verbose=1,
                            best_model_save_path=filename+'/',
                            log_path=filename+'/',
                            eval_freq=30000,
                            deterministic=True,
                            render=False
                            )

update_callback = UpdateDesiredCallback(train_env)
callback = CallbackList([eval_callback, update_callback])
oa_model.learn(total_timesteps=5000000, #int(8e6),
            callback=callback,
            log_interval=100,
           )

oa_model.save(filename+'/success_model.zip')
print(filename)
