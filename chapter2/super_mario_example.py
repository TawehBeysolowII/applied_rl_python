#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:00:57 2019

@author: tawehbeysolow
"""

import tensorflow as tf
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from algorithms.actor_critic_utilities import Model
from neural_networks.models import ActorCriticModel

environment = gym_super_mario_bros.make('SuperMarioBros-v0')
environment = BinarySpaceToDiscreteSpaceEnv(environment, SIMPLE_MOVEMENT)
observation = environment.reset()

def play_super_mario(policy_model=ActorCriticModel, environment=environment):
    
    observation_space = environment.observation_space
    action_space = environment.action_space
 
    with tf.Session(config=tf.ConfigProto()) as session:
        
        model = Model(session=session,
                      policy_model=policy_model,
                      observation_space=observation_space,
                      action_space=action_space,
                      n_environments=1,
                      n_steps=1,
                      entropy_coefficient=0,
                      value_coefficient=0,
                      max_grad_norm=0)
        
        #model.load("./models/260/model.ckpt")
        observations = environment.reset()
        score, n_step, done = 0, 0, False
    
        while not done:
            
            actions, values = model.step(observations)
            
            for action in actions:
                
                observations, rewards, done, info = environment.step(action)
                                
                score += rewards
            
                environment.render()
                
                n_step += 1
                        
        print('Step: %s \nScore: %s '%(n_step, score))
        environment.reset()

if __name__ == '__main__':
    
    
    play_super_mario()