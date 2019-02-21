#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:16:24 2019

@author: tawehbeysolow
"""

from pyglet.window import key
import numpy as np
from gym.envs.box2d.car_racing import CarRacing

def car_racing(record=False):
    environment = CarRacing()
    environment.reset()
    action = np.array([1.0, 1.0, 1.0])
    total_reward = 0.0
    steps = 0
    restart = False
   
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  action[0] = -1.0
        if k==key.RIGHT: action[0] = +1.0
        if k==key.UP:    action[1] = +1.0
        if k==key.DOWN:  action[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        
    def key_release(k, mod):
        if k==key.LEFT  and action[0]==-1.0: action[0] = 0
        if k==key.RIGHT and action[0]==+1.0: action[0] = 0
        if k==key.UP:    action[1] = 0
        if k==key.DOWN:  action[2] = 0
        
    for _ in range(1000):
        environment.render()
        environment.viewer.window.on_key_press = key_press
        environment.viewer.window.on_key_release = key_release
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1            
        observation, reward, done, info = environment.step(action)
        print("Step {}:".format(_))
        print("action: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))
    
    
if __name__ == '__main__':
    
    car_racing()