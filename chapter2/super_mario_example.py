#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:00:57 2019

@author: tawehbeysolow
"""

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
environment = gym_super_mario_bros.make('SuperMarioBros-v0')
environment = BinarySpaceToDiscreteSpaceEnv(environment, SIMPLE_MOVEMENT)
observation = environment.reset()


for step in range(5000):
    observation, reward, done, info = environment.step(environment.action_space.sample())
    environment.render()

environment.close()