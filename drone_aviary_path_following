import tensorflow as tf
import os
import time
from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from gym import spaces
from torch.autograd import Variable
from collections import deque
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.noise import NormalActionNoise

from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary



"""## **Creating the Environment**"""

class DroneAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=250,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 randomize_init=False
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )
        self.EPISODE_LEN_SEC = 52
        self.desPos = np.array([0.,0.,1.])
        self.P1 = np.array([0,0,0])     #Penalty parameters multiplied by position error
        self.P2 = np.array([0,0,0])     #Penalty parameters multiplied by angular error
        self.P3 = 0
        self.P4 = 0                     #Penalty parameters multiplied by velocity error
        self.maxX = 10.
        self.minX = -10.
        self.maxY = 10.
        self.minY = -10.
        self.maxZ = 20.
        self.minZ = 0.08
        self.lastActionNorm = np.array([0.,0.,0.,0.])
        self.currentActionNorm = np.array([0.,0.,0.,0.])
        self.randomize_init = randomize_init

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([-1.,           -1.,           -1.,           -1.])
        act_upper_bound = np.array([ 1.,            1.,            1.,            1.])
        total_actions = spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
                                               high=act_upper_bound,
                                               dtype=np.float32
                                               ) for i in range(self.NUM_DRONES)})
        return total_actions["0"]
    
    ################################################################################

    ################################################################################

    def randInit(self):
        """Update the Desired Position.

        Returns
        -------
        Nothing

        """

        meanX = (self.maxX + self.minX) / 2 
        stdX = (self.maxX - self.minX) * 0.3
        meanY = (self.maxY + self.minY) / 2 
        stdY = (self.maxY - self.minY) * 0.3
        meanZ = (self.maxZ + self.minZ) / 2 
        stdZ = (self.maxZ - self.minZ) * 0.3

        self.INIT_XYZS[0][0] = np.random.normal(meanX,stdX,1).clip(0.99*self.minX,0.99*self.maxX)
        self.INIT_XYZS[0][1] = np.random.normal(meanY,stdY,1).clip(0.99*self.minY,0.99*self.maxY)
        self.INIT_XYZS[0][2] = np.random.normal(meanZ,stdZ,1).clip(0.1125,0.99*self.maxZ)
        self.INIT_RPYS[0] = np.random.normal(0,4 * (np.pi/180),3).clip(-8 * (np.pi/180),8 * (np.pi/180))
    
    ################################################################################


    ################################################################################

    def updateBound(self,bound):
        """Update the Bounderies of the learning Zone.

        Returns
        -------
        Nothing

        """
        #bound =[max x   min x   max y   min y   max z   min z]

        self.maxX, self.minX, self.maxY, self.minY, self.maxZ, self.minZ = bound[0:6]
    
    ################################################################################

    ################################################################################

    def updatePar(self, P1, P2, P3, P4):
        """Update the Desired Position.

        Returns
        -------
        Nothing

        """
        self.P1[0:3] = P1[0:3]
        self.P2[0:3] = P2[0:3]
        self.P3 = P3
        self.P4 = P4
    
    ################################################################################

    ################################################################################

    def randDes(self):
        """Update the Desired Position.

        Returns
        -------
        Nothing

        """
        meanX = (self.maxX + self.minX) / 2 
        stdX = (self.maxX - self.minX) * 0.3
        meanY = (self.maxY + self.minY) / 2 
        stdY = (self.maxY - self.minY) * 0.3
        meanZ = (self.maxZ + self.minZ) / 2 
        stdZ = (self.maxZ - self.minZ) * 0.3

        desPos = np.array([0.,0.,0.])
        desPos[0] = np.random.normal(meanX,stdX,1).clip(0.99*self.minX,0.99*self.maxX)
        desPos[1] = np.random.normal(meanY,stdY,1).clip(0.99*self.minY,0.99*self.maxY)
        desPos[2] = np.random.normal(meanZ,stdZ,1).clip(0.1125,0.99*self.maxZ)

        self.updateDes(desPos)
    
    ################################################################################

    ################################################################################

    def generateRand(self):
        """Return Random initial and desired

        Returns
        -------
        Nothing

        """
        meanX = (self.maxX + self.minX) / 2 
        stdX = (self.maxX - self.minX) * 0.3
        meanY = (self.maxY + self.minY) / 2 
        stdY = (self.maxY - self.minY) * 0.3
        meanZ = (self.maxZ + self.minZ) / 2 
        stdZ = (self.maxZ - self.minZ) * 0.3

        randDes = np.zeros([10,3])
        randInitPos = np.zeros([10,3])
        randInitAng = np.zeros([10,3])

        for i in range(np.shape(randInitAng)[0]):
          randInitAng[i] = np.random.normal(0,4 * (np.pi/180),3).clip(-8 * (np.pi/180),8 * (np.pi/180))
          randDes[i][0] = np.random.normal(meanX,stdX,1).clip(0.99*self.minX,0.99*self.maxX)
          randDes[i][1] = np.random.normal(meanY,stdY,1).clip(0.99*self.minY,0.99*self.maxY)
          randDes[i][2] = np.random.normal(meanZ,stdZ,1).clip(0.1125,0.99*self.maxZ)
          randInitPos[i][0] = np.random.normal(meanX,stdX,1).clip(0.99*self.minX,0.99*self.maxX)
          randInitPos[i][1] = np.random.normal(meanY,stdY,1).clip(0.99*self.minY,0.99*self.maxY)
          randInitPos[i][2] = np.random.normal(meanZ,stdZ,1).clip(0.1125,0.99*self.maxZ)

        return randDes, randInitPos, randInitAng
    ################################################################################


    ################################################################################

    def updateDes(self, desPos):
        """Update the Desired Position.

        Returns
        -------
        Nothing

        """
        self.desPos[0:3] = desPos[0:3]
    
    ################################################################################

    ################################################################################

    def updateInit(self, initPos, initAng):
        """Update the Desired Position.

        Returns
        -------
        Nothing

        """
        self.INIT_XYZS[0][0:3] = initPos[0:3]
        self.INIT_RPYS[0][0:3] = initAng[0:3]

    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        total_states = spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                     high=obs_upper_bound,
                                                                     dtype=np.float32
                                                                     ),
                                                 "neighbors": spaces.MultiBinary(self.NUM_DRONES)
                                                 }) for i in range(self.NUM_DRONES)})
        """
        
        #### Observation vector ### X        Y        Z         R       P       Y       VX       VY       VZ       WX       WY       WZ        DX         DY        DZ
        # obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,  -np.inf,  -np.inf])
        # obs_upper_bound = np.array([ np.inf,  np.inf,  np.inf, np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,   np.inf,   np.inf])
        # total_states = spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
        #                                                             high=obs_upper_bound,
        #                                                             dtype=np.float32
        #                                                             ),
        #                                         "neighbors": spaces.MultiBinary(self.NUM_DRONES)
        #                                         }) for i in range(self.NUM_DRONES)})
        # return total_states["0"]["state"]
        
        #### Observation vector ###   R       P       Y       VX       VY       VZ       WX       WY       WZ        DX         DY        DZ       X         Y         Z
        obs_lower_bound = np.array([-np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf])
        obs_upper_bound = np.array([ np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf])
        total_states = spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                      high=obs_upper_bound,
                                                                      dtype=np.float32
                                                                      ),
                                                  "neighbors": spaces.MultiBinary(self.NUM_DRONES)
                                                  }) for i in range(self.NUM_DRONES)})
        return total_states["0"]["state"]

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        total_obs = {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i, :]} for i in range(self.NUM_DRONES)}
        obs = total_obs["0"]["state"]
        errPos = obs[0:3] - self.desPos[0:3]
        return np.hstack([obs[7:10], obs[10:13], obs[13:16],  errPos[0:3], obs[0:3]]).reshape(15,)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        #clipped_action = np.zeros((self.NUM_DRONES, 4))
        #for k, v in action.items():
        #    clipped_action[int(k), :] = np.clip(np.array(v), 0, self.MAX_RPM)
        #return clipped_action
        
        #return np.array((self.MAX_RPM / 2) * (1 + action))
        return np.array(self.HOVER_RPM * (1+0.15*action))

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        state = self._getDroneStateVector(0)
        x,y,z = state[0:3]
        r,p,y = state[7:10]
        self.currentActionNorm = ((state[16:20] / self.HOVER_RPM) - 1) / 0.15

        accErr = self.P3 * np.sum(np.square(self.currentActionNorm - self.lastActionNorm))
        hoverErr = self.P4 * np.sum(np.square(self.currentActionNorm))
        #posErr = np.sum(self.P1 * np.square(self.desPos-state[0:3]))
        #angErr = np.sum(self.P2 * np.square(state[7:10]))
        #reward = np.tanh(1 - posErr - angErr) - accErr - hoverErr


        posErr = np.sqrt(np.sum(np.square(self.desPos-state[0:3])))
        stdPos = 0.5
        a = 7

        if posErr <= 0.03:
          posErrFun = 10

        else:
          posErrFun = (a / np.sqrt(2 * np.pi * (stdPos**2))) * (np.exp(-0.5 * ((posErr / stdPos)**2))) + (1 / (a * posErr))

        reward = posErrFun
        self.lastActionNorm = self.currentActionNorm

        #if (r > np.pi/2 or r < -np.pi/2) or (p > np.pi/2 or p < -np.pi/2) or (y > np.pi/2 or y < -np.pi/2):
        #  reward = reward - 7500
        #elif (x > self.maxX or x < self.minX) or (y > self.maxY or y < self.minY) or (z > self.maxZ or z < self.minZ):
        #  reward = reward - 7500

        return reward

        #if reward > 0:
        #  return reward
        #else:
        #  return 0

        #return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        state = self._getDroneStateVector(0)

        x,y,z = state[0:3]
        roll,pitch,yaw = state[7:10]

        if (x > self.maxX or x < self.minX) or (y > self.maxY or y < self.minY) or (z > self.maxZ or z < self.minZ):
          if self.randomize_init:
            self.randInit()
          return True
        elif (roll > np.pi/2 or roll < -np.pi/2) or (pitch > np.pi/2 or pitch < -np.pi/2) or (yaw > np.pi/2 or yaw < -np.pi/2):
          if self.randomize_init:
            self.randInit()
          return True
        elif self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
          if self.randomize_init:
            self.randInit()
          return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

"""# **Callback Functions**

## Update Desired Callback
"""
