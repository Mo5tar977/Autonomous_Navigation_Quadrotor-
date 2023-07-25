
"""## **Importing Library**"""

from drone_aviary_path_following import *

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
        if self.episodeCount % 5 == 0:
          self.train_env.randInit()
          #print("Upadated Initial After: ",self.num_timesteps," Timesteps")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

"""# **Main Code**

## **Parameters**
"""

P1 = np.array([20,20,30])
P2 = np.array([0,0,0])
P3 = 0
P4 = 0

"""## **Train Environment**"""

filename = 'results/save-'+"DroneAviary"+'-'+"TD3-15-state-fixed-desired"+'-'+"Kin"+'-'+"RPM"+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
if not os.path.exists(filename):
  print("True")
  os.makedirs("./"+ filename)
print(filename)
train_env = DroneAviary()
train_env.updatePar(P1,P2,P3,P4)
print("[INFO] Action space:", train_env.action_space)
print("[INFO] Observation space:", train_env.observation_space)

"""## **TD3 Algorithm**"""

offpolicy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                        net_arch=dict(qf=[400, 300], pi=[400, 300]))
model = TD3(td3ddpgMlpPolicy,
                    train_env,
                    learning_rate=0.0007,
                    action_noise=NormalActionNoise(0,0.2),
                    learning_starts=1e4,
                    batch_size=256,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    verbose=1
                    )


"""## **Evaluation Environment**"""

eval_env = DroneAviary(randomize_init=True)
eval_env.updatePar(P1,P2,P3,P4)
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
model.learn(total_timesteps=5000000, #int(5e6),
            callback=callback,
            log_interval=100,
           )

"""## **Model**"""

#### Save the model ########################################
model.save(filename+'/success_model.zip')
print(filename)

    #### Print training progression ############################
#with np.load(filename+'/evaluations.npz') as data:
#    for j in range(data['timesteps'].shape[0]):
#        print(str(data['timesteps'][j])+","+str(data['results'][j][0][0]))

