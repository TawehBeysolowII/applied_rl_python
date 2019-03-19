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
from baselines.common.distributions import make_pdtype

class activation_dictionary():
    
    def __init__(self, activation):
        
        dictionary = {'elu': tf.nn.elu,
                      'relu': tf.nn.relu, 
                      'selu': tf.nn.selu, 
                      'sigmoid': tf.nn.sigmoid,
                      'softmax': tf.nn.softmax,
                       None: None}
        
        return dictionary[activation]

def normalized_columns_initializer(standard_deviation=1.0):
  def initializer(shape, dtype=None, partition_info=None):
    output = np.random.randn(*shape).astype(np.float32)
    output *= standard_deviation/float(np.sqrt(np.square(output).sum(axis=0, keepdims=True)))
    return tf.constant(output)
  return initializer

def linear_operation(x, size, name, initializer=None, bias_init=0):
  with tf.variable_scope(name):
    weights = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer)
    biases = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, weights) + biases

def convolution_layer(inputs, filters, kernel_size, strides, activation='relu'):
    
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            strides=(strides, strides),
                            activation=activation_dictionary(activation=activation))


def fully_connected_layer(inputs, n_units, activation, gain=np.sqrt(2)):
    return tf.layers.dense(inputs=inputs, 
                           units=n_units, 
                           gain=gain,
                           activation=activation_dictionary(activation=activation),
                           kernel_initializer=tf.contrib.layers.xavier_initializer())

def lstm_layer(input, size, actions, apply_softmax=False):
      input = tf.expand_dims(input, [0])
      lstm = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True)
      state_size = lstm.state_size
      step_size = tf.shape(input)[:1]
      c_init = np.zeros((1, state_size.c), np.float32)
      h_init = np.zeros((1, state_size.h), np.float32)
      initial_state = [c_init, h_init]
      cell_state = tf.placeholder(tf.float32, [1, state_size.c])
      hidden_state = tf.placeholder(tf.float32, [1, state_size.h])
      input_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
      
      _outputs, states = tf.nn.dynamic_rnn(cell=lstm,
                                           inupts=input,
                                           initial_state=input_state,
                                           sequence_length=step_size,
                                           time_major=False)
      _cell_state, _hidden_state = states
      output = tf.reshape(_outputs, [-1, size])
      output_state = [_cell_state[:1, :], _hidden_state[:1, :]]
      output = linear_operation(output, actions, "logits", normalized_columns_initializer(0.01))
      output = tf.nn.softmax(output, dim=-1)
      return output, input_state, output_state, initial_state

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


class DeepQNetwork():
    
    def __init__(self, n_units, n_classes, n_filters, stride, kernel, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.stride = stride
        self.kernel = kernel
        
        input_matrix = tf.placeholder(tf.float32, [None, state_size])
        actions = tf.placeholder(tf.float32, [None, n_classes])
        
        target_Q = tf.placeholder(tf.float32, [None])
        
        network1 = convolution_layer(inputs=input_matrix, 
                                     filters=self.n_filters, 
                                     kernel_size=self.kernel, 
                                     strides=self.stride,
                                     activation='elu')
        
        network1 = tf.layers.batch_normalization(self.network1,
                                                 training=True,
                                                 epsilon=1e-5)    

        network2 = convolution_layer(inputs=network1, 
                                     filters=self.n_filters*2, 
                                     kernel_size=self.kernel/float(2), 
                                     strides=self.stride/float(2), 
                                     activation='elu')
     
        network2 = tf.layers.batch_normalization(inputs=network2,
                                                 training=True,
                                                 epsilon=1e-5)

        network3 = convolution_layer(inputs=network1, 
                                     filters=self.n_filters*4, 
                                     kernel_size=self.kernel/float(2), 
                                     strides=self.stride/float(2),
                                     activation='elu')
     
        network3 = tf.layers.batch_normalization(inputs=network3,
                                                 training=True,
                                                 epsilon=1e-5)

        network3 = tf.layers.flatten(inputs=network3)
        
        output = fully_connected_layer(inputs=network3, 
                                       units=self.n_units,
                                       activation='elu')
        
        output = fully_connected_layer(inputs=output,
                                       units=n_classes,
                                       activation=None)
        
        predicted_Q = tf.reduce_sum(tf.multiply(output, actions), axis=1)
        
        self.error_rate = tf.reduce_mean(tf.square(target_Q - predicted_Q))
        
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.error_rate)
 

class ActorCriticModel():
    
    def __init__(self, session, environment, action_space, n_batches, n_steps, reuse=False):

        self.distribution_type = make_pdtype(action_space)
        height, weight, channel = environment.shape
        environment_shape = (height, weight, channel)
        inputs_ = tf.placeholder(tf.float32, [None, environment_shape], name="input")

        scaled_images = tf.cast(inputs_, tf.float32)/float(255)
        
        
        layer1 = tf.layers.batch_normalization(convolution_layer(inputs=scaled_images, 
                                                                 filters=32, 
                                                                 kernel_size=8, 
                                                                 strides=4, 
                                                                 gain=np.sqrt(2)))
                    
        layer2 = tf.layers.batch_normalization(convolution_layer(inputs=tf.nn.relu(layer1), 
                                                                 units=64, 
                                                                 kernel_size=4, 
                                                                 strides=2, 
                                                                 gain=np.sqrt(2)))
        
        layer3 = tf.layers.batch_normalization(convolution_layer(inputs=tf.nn.relu(layer2), 
                                                                 units=64, 
                                                                 kernel_size=3, 
                                                                 strides=1,
                                                                 gain=np.sqrt(2)))
        
        layer4 = tf.layers.flatten(inuts=layer3)
        
        output_layer = tf.nn.softmax(fully_connected_layer(inputs=layer4, 
                                                           units=512, 
                                                           gain=np.sqrt(2)))

        self.distribution, self.pi = self.pdtype.pdfromlatent(output_layer, init_scale=0.01)

        value_of_state = fully_connected_layer(output_layer, 1, activation=None)[:, 0]

        self.initial_state = None

        sampled_action = self.distribution_type.sample()

        def step(current_state, *_args, **_kwargs):
            action, value = session.run([sampled_action, value_of_state], {inputs_: current_state})
            return action, value

        def value(current_state, *_args, **_kwargs):
            return session.run(value_of_state, {inputs_: current_state})

        def select_action(current_state, *_args, **_kwargs):
            return session.run(sampled_action, {inputs_: current_state})

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
        
        def log_likelihood_loss(actual_labels, predicted_labels):
            log_likelihood = backend.log(actual_labels * (actual_labels - predicted_labels) + 
                                  (1 - actual_labels) * (actual_labels + predicted_labels))
            return backend.mean(log_likelihood * advantages, keepdims=True)
        
        if self.loss_function == 'log_likelihood':
            self.loss_function = log_likelihood_loss
                
        policy_model = Model(inputs=[input_layer, advantages], outputs=output_layer)
        policy_model.compile(loss=self.loss_function, optimizer=Adam(self.learning_rate))
        model_prediction = Model(input=[input_layer], outputs=output_layer)
        return policy_model, model_prediction



        