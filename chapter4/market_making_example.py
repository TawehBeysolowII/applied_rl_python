#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:00:05 2019

@author: tawehbeysolow
"""

import random, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tgym.envs import SpreadTrading
from neural_networks.market_making_models import DeepQNetworkMM, Memory
from tgym.gens.deterministic import WavySignal

#Parameters
episodes = 50
trading_fee = .2
time_fee = 0
history_length = 2
memory_size = 3000
gamma = 0.96
epsilon_min = 0.01
batch_size = 64
action_size = len(SpreadTrading._actions)
train_interval = 10
learning_rate = 0.001
n_layers = 2
n_units = 500
n_classes = 3
goal = 190
n_episodes = 400
max_steps = 100
explore_start = 1.0
explore_stop = 0.01
decay_rate = 1e-4
hold =  np.array([1, 0, 0])
buy = np.array([0, 1, 0])
sell = np.array([0, 0, 1])
possible_actions = [hold, buy, sell]

#Classes and variables
memory = Memory(max_size=memory_size)

generator = WavySignal(period_1=25, period_2=50, epsilon=-0.5)


environment = SpreadTrading(spread_coefficients=[1],
                            data_generator=generator,
                            trading_fee=trading_fee,
                            time_fee=time_fee,
                            history_length=history_length)

state_size = len(environment.reset())


def exploit_explore(session, model, explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)
        
    else:
        Qs = session.run(model.output_layer, feed_dict = {model.input_matrix: state.reshape((1, state.shape[0]))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability


def train_model(model, environment):
    tf.summary.scalar('Loss', model.error_rate)
    scores = []
    done = False
    error_rate = 0
    
    states = np.empty(0).reshape(0, state_size)
    actions = np.empty(0).reshape(0, 3)
    rewards = np.empty(0).reshape(0, 1)
    observation = environment.reset()

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        decay_step = 0

        for episode in range(n_episodes):
            
            step, reward_sum = 0, []
            state = np.reshape(observation, [1, state_size])        

            while step < max_steps:
                
                step += 1; decay_step += 1
                
                action, explore_probability = exploit_explore(session=sess,
                                                              model=model,
                                                              explore_start=explore_start, 
                                                              explore_stop=explore_stop, 
                                                              decay_rate=decay_rate, 
                                                              decay_step=decay_step, 
                                                              state=state, 
                                                              actions=possible_actions)
                
                state, reward, done, info = environment.step(action)
                
                reward_sum.append(reward)
                
                if 'status' in info and info['status'] == 'Closed plot' or step >= max_steps:
                    done = True
                else:
                    environment.render()

                if done:
                    
                    next_state = np.zeros((state_size,), dtype=np.int)
                    step = max_steps                    
                    total_reward = np.sum(reward_sum)                    
                    scores.append(total_reward)                    
                    memory.add((state, action, reward, next_state, done))
                   
                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Loss: {}'.format(error_rate),
                              'Explore P: {:.4f}'.format(explore_probability))

                else:
                    next_state = environment.reset()
                    state = next_state
                    memory.add((state, action, reward, next_state, done))

                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch]) 
                next_states = np.array([each[3] for each in batch])
                dones = np.array([each[4] for each in batch])

                target_Qs_batch = []
                
                Qs_next_state = sess.run(model.predicted_Q, feed_dict={model.input_matrix: next_states, model.actions: actions})
                
                for i in range(0, len(batch)):
                    terminal = dones[i]

                    if terminal:
                        target_Qs_batch.append(rewards[i])
                        
                    else:
                        target = rewards[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        
                    
                targets = np.array([each for each in target_Qs_batch])

                error_rate, _ = sess.run([model.error_rate, model.optimizer], 
                                          feed_dict={model.input_matrix: states,
                                                     model.target_Q: targets,
                                                     model.actions: actions})

                '''
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={model.inputs_: states,
                                                   model.target_Q: targets,
                                                   model.actions_: actions})

                writer.add_summary(summary, episode)
                writer.flush()
              

            if episode % 5 == 0:
                #saver.save(sess, filepath+'/models/model.ckpt')
                #print("Model Saved")
                '''
    
            
if __name__ == '__main__':
    
    model = DeepQNetworkMM(n_units=n_units, 
                           n_classes=n_classes, 
                           state_size=state_size, 
                           action_size=action_size, 
                           learning_rate=learning_rate)
    
    trained_model = train_model(model=model,
                                environment=environment)
