#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:16:38 2019

@author: tawehbeysolow
"""

import numpy as np 
from collections import defaultdict


def toDiscreteStates(observation):
	interval=[0 for i in range(len(observation))]
	max_range=[1.2,0.07]	#[4.8,3.4*(10**38),0.42,3.4*(10**38)]

	for i in range(len(observation)):
		data = observation[i]
		inter = int(math.floor((data + max_range[i])/(2*max_range[i]/buckets[i])))
		if inter>=buckets[i]:
			interval[i]=buckets[i]-1
		elif inter<0:
			interval[i]=0
		else:
			interval[i]=inter
	return interval

def get_action(observation,t):

	if np.random.random()<max(0.001, min(0.015, 1.0 - math.log10((t+1)/220.))):#get_epsilon(t):
		return env.action_space.sample()
	interval = toDiscreteStates(observation)
	
	# if Q_table[tuple(interval)][0] >=Q_table[tuple(interval)][1]:
	# 	return 0
	# else:
	# 	return 1
	return np.argmax(np.array(Q_table[tuple(interval)]))

def updateQ_SARSA(observation,reward,action,ini_obs,next_action,t):
	
	interval = toDiscreteStates(observation)

	Q_next = Q_table[tuple(interval)][next_action]

	ini_interval = toDiscreteStates(ini_obs)

	Q_table[tuple(ini_interval)][action]+=max(0.4, min(0.1, 1.0 - math.log10((t+1)/125.)))*(reward + gamma*(Q_next) - Q_table[tuple(ini_interval)][action])

