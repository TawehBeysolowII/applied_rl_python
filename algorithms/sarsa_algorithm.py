#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:16:38 2019

@author: tawehbeysolow
"""

import numpy as np 
from collections import defaultdict


def greedy_action_policy(Q, observation, epsilon=0.1, nA=2):
    
    A = np.ones(nA, dtype=float)*epsilon/nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A
    

def sarsa_algorithm(environment, n_episodes):
    
    q_dictionary = defaultdict(lambda: np.zeros(environment.action_space.n))
    
    for _ in range(n_episodes):
        
        current_state = environment.reset()
        probabilities = greedy_action_police(Q, observation)
        
        