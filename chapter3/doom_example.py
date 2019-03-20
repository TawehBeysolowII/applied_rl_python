#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:50:31 2019

@author: tawehbeysolow
"""

import warnings, random, time, tensorflow as tf, numpy as np, matplotlib.pyplot as plt  
from neural_networks.models import DeepQNetwork
from algorithms.dql_utilities import create_environment, stack_frames, Memory

#Parameters
gamma = 0.95
memory_size = int(1e7)
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
action_size = 3 #environment.get_avaiable_buttons_size()
explore_start = 1.0
explore_stop = 0.01
decay_rate = 1e-4
pretrain_length = batch_size
warnings.filterwarnings('ignore')
writer = tf.summary.FileWriter("/tensorboard/dqn/1")
write_op = tf.summary.merge_all()

def play_doom(model, environment, train=True):
    tf.summary.scalar('Loss', model.loss)
    saver = tf.train.Saver()
    stacked_frames = list()
    memory = Memory(max_size=memory_size)
    
    if train == True:
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            decay_step = 0
            environment.init()
    
            for episode in range(n_episodes):
                step, reward_sum = 0, []
                environment.new_episode()
                state = environment.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, state, True)
    
                while step < max_steps:
                    step += 1; decay_step +=1
                    action, explore_probability = predict_action(explore_start, 
                                                                 explore_stop, 
                                                                 decay_rate, 
                                                                 decay_step, 
                                                                 state, 
                                                                 possible_actions)
    
                    reward = environment.make_action(action)
                    done = environment.is_episode_finished()
                    reward_sum.append(reward)
    
                    if done:
                        next_state = np.zeros((84,84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        step = max_steps
                        total_reward = np.sum(reward_sum)
    
                        print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Training loss: {:.4f}'.format(loss),
                                  'Explore P: {:.4f}'.format(explore_probability))
    
                        memory.add((state, action, reward, next_state, done))
    
                    else:
                        next_state = environment.get_state().screen_buffer
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        memory.add((state, action, reward, next_state, done))
                        state = next_state
    
    
                    batch = memory.sample(batch_size)
                    states = np.array([each[0] for each in batch], ndmin=3)
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch]) 
                    next_states = np.array([each[3] for each in batch], ndmin=3)
                    dones = np.array([each[4] for each in batch])
    
                    target_Qs_batch = []
    
                     # Get Q values for next_state 
                    Qs_next_state = sess.run(DeepQNetwork.output, feed_dict = {DeepQNetwork.inputs_: next_states})
                    
                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones[i]
    
                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards[i])
                            
                        else:
                            target = rewards[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)
                            
    
                    targets = np.array([each for each in target_Qs_batch])
    
                    loss, _ = sess.run([model.loss, model.optimizer],
                                        feed_dict={model.inputs_: states,
                                                   model.target_Q: targets,
                                                   model.actions_: actions})
    
                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={model.inputs_: states,
                                                       model.target_Q: targets,
                                                       model.actions_: actions})
    
                    writer.add_summary(summary, episode)
                    writer.flush()
    
                # Save model every 5 episodes
                if episode % 5 == 0:
                    #saver.save(sess, filepath+'/models/model.ckpt')
                    print("Model Saved")

def doom_example(model, environment):
    
    stacked_frames = list()
    environment.new_episode()
    
    for _ in range(n_episodes):
    
        for i in range(pretrain_length):
            
            if i == 0:
                state = environment.get_state().screen_buffer
                state, stacked_frames = stack_frames(stacked_frames, state, True)
            
            action = random.choice(possible_actions)
            reward = environment.make_action(action)
            done = environment.is_episode_finished()
            time.sleep(0.010)
            
            if done:
                next_state = np.zeros(state.shape)            
                memory.add((state, action, reward, next_state, done))
                time.sleep(3)
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
    
    
    
    
    