#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:50:31 2019

@author: tawehbeysolow
"""

import random, time, tensorflow as tf, numpy as np, matplotlib.pyplot as plt  
from neural_networks.models import DeepQNetwork
from algorithms.dql_utilities import create_environment, stack_frames, Memory

#Parameters
gamma = 0.95
memory_size = 1e7
train = True
episode_render = False
n_units = 500
n_classes = 3
learning_rate = 1e-4
stride = 4 
kernel = 8
n_filters = 3
n_episodes = 500
max_steps = 100
batch_size = 50
environment, possible_actions = create_environment()
state_size = [84, 84, 4]
action_size = environment.get_avaiable_buttons_size()
explore_start = 1.0
explore_stop = 0.01
decay_rate = 1e-4


def doom_example(model, environment):
    
    stacked_frames = list()
    memory = Memory(max_size=memory_size)
    environment.new_episode()
    
    for i in range(pretrain_length):
        
        if i == 0:
            state = environment.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        action = random.choice(possible_actions)
        reward = environment.make_action(action)
        done = environment.is_episode_finished()
        
        if done:
            next_state = np.zeros(state.shape)            
            memory.add((state, action, reward, next_state, done))
            environment.new_episode()
            state = environment.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        else:
            next_state = environment.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            memory.add((state, action, reward, next_state, done))
            state = next_state
    
    
if __name__ == '__main__':
    
    
    model = DeepQNetwork(n_units=n_units, 
                         n_classes=n_classes, 
                         n_filters=n_filters, 
                         stride=stride, 
                         kernel=kernel, 
                         state_size=state_size, 
                         action_size=action_size, 
                         learning_rate=learning_rate)
    
    
    doom_example(model=model,
                environment=environment)
    
    
    
    
    