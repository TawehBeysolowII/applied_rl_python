#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:50:58 2019

@author: tawehbeysolow
"""

import gym, numpy as np
from neural_networks.models import MLPModelKeras

#Parameters 
n_units = 8
gamma = .99
batch_size = 50
learning_rate = 1e-4
n_episodes = 10000
render = False
goal = 190
n_layers = 2
n_classes = 2
environment = gym.make('CartPole-v1')

def cart_pole_game(environment, policy_model, model_predictions):
    loss = []
    n_episode, reward_sum, done = 0, 0, False
    n_actions = environment.action_space.n
    observation = environment.reset()
    
    states = np.empty(0).reshape(0, len(environment.reset()))
    actions = np.empty(0).reshape(0, 1)
    rewards = np.empty(0).reshape(0, 1)
    discounted_rewards = np.empty(0).reshape(0, 1)
    
    
    while n_episode < n_episodes:
        
        state = np.reshape(observation, [1, len(environment.reset())])
        prediction = model_predictions.predict([state])[0]
        action = np.random.choice(range(n_actions), p=prediction)
        
        #Appending the observations and outputs 
        states, actions = np.vstack([states, state]), np.vstack([actions, action])
        
        observation, reward, done, info = environment.step(action)
        rewards = np.vstack([rewards, reward])
        reward_sum += reward 


        if done == True:
            
            discounted_reward = calculate_discounted_reward(rewards)
            discounted_rewards = np.vstack([discounted_rewards, discounted_reward])
            
            
            if n_episodes+1%batch_size == 0:
                
                discounted_rewards -= discounted_rewards.mean()
                discounted_rewards /= discounted_rewards.std()
                discounted_rewards = discounted_rewards.squeeze()
                actions = actions.squeeze().astype(int)
                
                
                train_actions = np.zeros([len(actions), n_actions])
                train_actions[np.arange(len(actions)), actions] = 1
                
                loss.append(policy_model.train_on_batch([states, discounted_rewards], train_actions))
                
                
                #Resetting variables 
                states = np.empty(0).reshape(0, len(environment))
                actions = np.empty(0).reshape(0, 1)
                discounted_rewards = np.emtpy(0).reshape(0, 1)
                
                
            if n_episodes+1%batch_size == 0:
                
                score = score_model(model_predictions, 10)
                
                print('''Episode: %s \nAverage Reward: %s 
                      \nScore: %s \nError: %s''')%(n_episode+1, reward_sum/float(batch_size), score, np.mean(loss[-batch_size:]))
            




def calculate_discounted_reward(reward, gamma=gamma):
    output, prior = [], 0    
    for _reward in reward:
        output.append(_reward + prior * gamma)
        prior = _reward    
    return output[::-1]


def score_model(model, n_tests, render=False):
    scores = []    
    for _ in range(n_tests):
        environment.reset()
        observation = environment
        reward_sum = 0
        while True:
            if render:
                environment.render()

            state = np.reshape(observation, [1, len(environment.reset())])
            predict = model.predict([state])[0]
            action = np.argmax(predict)
            observation, reward, done, _ = environment.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
        
    environment.close()
    return np.mean(scores)


if __name__ == '__main__':
    
    
    mlp_model = MLPModelKeras(n_units=n_units, 
                              n_layers=n_layers, 
                              n_columns=len(environment.reset()), 
                              n_outputs=n_classes, 
                              learning_rate=learning_rate, 
                              hidden_activation='selu', 
                              output_activation='softmax')
    
    policy_model, model_predictions = mlp_model.get_policy_model(input_shape=(1, len(environment.reset())))
    
    policy_model.summary()
    
    cart_pole_game(environment=environment, 
                   policy_model=policy_model, 
                   model_predictions=model_predictions)



    
    
    
    
    
    