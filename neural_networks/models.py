#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:49:13 2019

@author: tawehbeysolow
"""

import tensorflow as tf, numpy as np
from sklearn.model_selection import train_test_split
import keras.layers as layers
from keras import backend
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
#from baselines.common.distributions import make_pdtype

class activation_dictionary():
    
    def __init__(self, activation):
        
        dictionary = {'relu': tf.nn.relu, 
                      'selu': tf.nn.selu, 
                      'sigmoid': tf.nn.sigmoid,
                      'softmax': tf.nn.softmax}
        
        return dictionary[activation]

def convolution_layer(inputs, filters, kernel_size, strides, activation='relu'):
    
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=activation_dictionary(activation=activation))


def fully_connected_layer(inputs, n_units, activation):
    return tf.layers.dense(inputs=inputs, units=n_units, activation=activation_dictionary(activation=activation))


def create_weights_biases(n_layers, n_units, n_columns, n_outputs):
    '''
    Creates dictionaries of variable length for differing neural network models
    
    Arguments 
    
    n_layers - int - number of layers 
    n_units - int - number of neurons within each individual layer
    n_columns - int - number of columns within dataset
    
    :return: dict (int), dict (int)
    '''
    weights, biases = {}, {}
    for i in range(n_layers):
        if i == 0: 
            weights['layer'+str(i)] = tf.Variable(tf.random_normal([n_columns, n_units]))
            biases['layer'+str(i)] = tf.Variable(tf.random_normal([n_columns]))
        elif i != 0 and i != n_layers-1:
            weights['layer'+str(i)] = tf.Variable(tf.random_normal([n_units, n_units]))
            biases['layer'+str(i)] = tf.Variable(tf.random_normal([n_units]))
        elif i != 0 and i == n_layers-1:
            weights['output_layer'] = tf.Variable(tf.random_normal([n_units, n_outputs]))
            biases['output_layer'] = tf.Variable(tf.random_normal([n_outputs]))
            
    return weights, biases

def create_input_output(input_dtype, output_dtype, n_columns, n_outputs):
    '''
    Create placeholder variables for tensorflow graph
    
    '''
    
    X = tf.placeholder(shape=(None, n_columns), dtype=input_dtype)
    Y = tf.placeholder(shape=(None, n_outputs), dtype=output_dtype)
    return X, Y
 

class ActorCriticModel():
    
    def __init__(self, session, environment, action_space, n_batches, n_steps, reuse=False):
                # This will use to initialize our kernels
        gain = np.sqrt(2)

        # Based on the action space, will select what probability distribution type
        # we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType
        # aka Diagonal Gaussian, 3D normal distribution
        self.pdtype = make_pdtype(action_space)

        height, weight, channel = environment.shape
        environment_shape = (height, weight, channel)
        inputs_ = tf.placeholder(tf.float32, [None, environment_shape], name="input")

        # Normalize the images
        scaled_images = tf.cast(inputs_, tf.float32) / 255.
        layer1 = convolution_layer(scaled_images, 32, 8, 4, gain)
        layer2 = convolution_layer(layer1, 64, 4, 2, gain)
        layer3 = convolution_layer(layer2, 64, 3, 1, gain)
        layer4= tf.layers.flatten(layer3)
        output_layer = fully_connected_layer(layer4, 512, gain=gain)

        # This build a fc connected layer that returns a probability distribution
        # over actions (self.pd) and our pi logits (self.pi).
        self.pd, self.pi = self.pdtype.pdfromlatent(output_layer, init_scale=0.01)

        # Calculate the v(s)
        value_of_state = fully_connected_layer(output_layer, 1, activation_fn=None)[:, 0]

        self.initial_state = None

        # Take an action in the action distribution (remember we are in a situation
        # of stochastic policy so we don't always take the action with the highest probability
        # for instance if we have 2 actions 0.7 and 0.3 we have 30% chance to take the second)
        a0 = self.pd.sample()

        # Function use to take a step returns action to take and V(s)
        def step(state_in, *_args, **_kwargs):
            action, value = session.run([a0, value_of_state], {inputs_: state_in})
           
            #print("step", action)
            
            return action, value

        # Function that calculates only the V(s)
        def value(state_in, *_args, **_kwargs):
            return session.run(value_of_state, {inputs_: state_in})

        # Function that output only the action to take
        def select_action(state_in, *_args, **_kwargs):
            return session.run(a0, {inputs_: state_in})

        self.inputs_ = inputs_
        self.value_of_state = value_of_state
        self.step = step
        self.value = value
        self.select_action = select_action
        
        
        
        
class MLPModelKeras():
    
    def __init__(self, n_units, n_layers, n_columns, n_outputs, learning_rate, hidden_activation, output_activation, loss_function):
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_outputs = n_outputs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def create_policy_model(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        advantages = layers.Input(shape=[1])
        
        hidden_layer = layers.Dense(units=self.n_units, 
                                    activation=self.hidden_activation,
                                    use_bias=False,
                                    kernel_initializer=glorot_uniform(seed=42))(input_layer)
        
        output_layer = layers.Dense(units=self.n_outputs, 
                                    activation=self.output_activation,
                                    use_bias=False,
                                    kernel_initializer=glorot_uniform(seed=42))(hidden_layer)
        
        def log_likelihood_loss(y_true, y_pred):
            log_lik = backend.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            return backend.mean(log_lik * advantages, keepdims=True)
        
        if self.loss_function == 'log_likelihood':
            self.loss_function = log_likelihood_loss
                
        policy_model = Model(inputs=[input_layer, advantages], outputs=output_layer)
        policy_model.compile(loss=self.loss_function, optimizer=Adam(self.learning_rate))
        model_prediction = Model(input=[input_layer], outputs=output_layer)
        return policy_model, model_prediction



        