#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:03:43 2019

@author: tawehbeysolow
"""

import gym, numpy as np, cv2
from retro_contest.local import make
from retro import make as make_retro 
from baselines.common.atari_wrappers import FrameStack 


#Parameters
cv2.ocl.setUseOpenCL(False)

class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
    - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        return frame


class ActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, environment):
        
        super(ActionsDiscretizer, self).__init__(environment)
        
        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 
                   'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], 
                   ['DOWN'], ['DOWN', 'B'], ['B']]
        
        self._actions = []

        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True
            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            array = np.array([False] * 12)
            for button in action:
                array[buttons.index(button)] = True
            self._actions.append(array)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, environment):
        super(AllowBacktracking, self).__init__(environment)
        self.current_reward = 0
        self.max_reward = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self.currrent_x = 0
        self.max_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        observation, reward, episode_done, info = self.environment.step(action)
        self.current_reward += reward
        reward = max(0, self.current_reward - self.max_reward)
        self.max_reward = max(self.max_reward, self.current_reward)
        return observation, reward, episode_done, info

def make_environment(environment_index):
    '''
    Create an environment for Super Mario
    '''

    level_dictionary = [
            
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}
        
    ]
    
    # Make the environment
    print(level_dictionary[environment_index]['game'])
    print(level_dictionary[environment_index]['state'])
    
    environment = make(game=level_dictionary[environment_index]['game'], 
               state=level_dictionary[environment_index]['state'], 
               bk2dir="./records")

    environment = ActionsDiscretizer(environment)
    environment = RewardScaler(environment)
    environment = PreprocessFrame(environment)
    environment = FrameStack(environment, 4)
    environment = AllowBacktracking(environment)
    return environment


def make_level(level_number):
    return make_environment(int(level_number))

def make_test_level_Green():
    return make_test()


def make_test(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2', record='./records'):
    """
    Create an environment with some standard wrappers.
    """

    environment = make_retro(game=game, 
                             state=game, 
                             record=record)

    environment = ActionsDiscretizer(environment)
    environment = RewardScaler(environment)
    environment = PreprocessFrame(environment)
    environment = FrameStack(environment, 4)
    environment = AllowBacktracking(environment)
    return environment
