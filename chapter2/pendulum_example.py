#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:18:44 2019

@author: tawehbeysolow
"""


import gym, numpy as np
from neural_networks.models import LSTMModelKeras

#Parameters 
n_units = 5
gamma = .99
batch_size = 50
learning_rate = 1e-3
n_episodes = 10000
render = False
goal = 190
n_layers = 2
n_classes = 2
environment = gym.make('Pendulum-v0')
environment_dimension = len(environment.reset())


def pendulum_game()

