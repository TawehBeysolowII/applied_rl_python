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
 

class MLPModelTF():
    
    def __init__(self, n_units, n_layers, n_columns, n_outputs, learning_rate, hidden_activation, output_activation):
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_outputs = n_outputs
        self.hidden_activation = hidden_activation
        self.output_actiation = output_activation
        self.learning_rate = learning_rate
        
    def create_variables_placeholders(self, input_dtype, output_dtype):
        
        X, Y = create_input_output(input_dtype=input_dtype,
                                   output_dtype=output_dtype,
                                   n_columns=self.n_columns,
                                   n_outputs=self.n_outputs)
        
        weights, biases = create_weights_biases(n_units=self.n_units,
                                                n_layers=self.n_layers,
                                                n_columns=self.n_columns,
                                                n_outputs=self.n_outputs)
        
        
        return X, Y, weights, biases
    
    
    def train_model(self, x_data, y_data, X, Y, weights, biases, epochs, batch_size=32, dropout=None):
        
        train_x, train_y, test_x, test_y = train_test_split(x_data, y_data)
        
        input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
        input_layer = tf.nn.sigmoid(input_layer)
        
        if dropout is not None: 
            input_layer = tf.nn.dropout(input_layer, dropout)

        for _ in range(len(weights)-1):
            
            if _ == 0 and len(weights) == 2:            
                hidden_layer = tf.add(tf.multiply(input_layer, weights['hidden1']), biases['hidden1'])
                hidden_layer = tf.nn.relu(hidden_layer)
                
                if dropout is not None: 
                    hidden_layer = tf.nn.dropout(hidden_layer, dropout)
                
                output_layer = tf.add(tf.multiply(hidden_layer, weights['output_layer']), biases['output_layer'])
        
        if int(biases['output_layer'].shape[0]) == 1:
           error = tf.reduce_sum(tf.pow(output_layer - Y, 2))/(len(train_x))
           optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(error)      
                       
        else:
            
            if int(biases['output_layer'].shape[0]) == 2:
                predictions = tf.nn.sigmoid(output_layer)
            else:
                predictions = tf.nn.softmax(output_layer)
                
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1)), tf.float32))
            cross_entropy = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y), tf.float32))
            
            
            with tf.Session() as sess:
                
                sess.run(tf.initialize_global_variables())
                
                for epoch in range(epochs): #Cross-validating data
                    
                    rows = np.random.random_integers(0, len(train_x)-1, len(train_x)-1)
                    _train_x, _train_y = train_x[rows], train_y[rows]
                    
                    #Batch training
                    for start, end in zip(range(0, len(train_x)-1, batch_size), 
                                          range(batch_size, len(train_x)-1, batch_size)):
                        
                        __train_x, __train_y = _train_x[start:end], _train_y[start:end]
                        
                        if int(biases['output_layer'].shape[0]) > 1:
                            
                            _cross_entropy, _accuracy, _adam_optimizer = sess.run([cross_entropy, accuracy, optimizer],
                                                                     feed_dict={X:__train_x, Y:__train_y})
                            
                            if epoch%10 == 0 and epoch > 0:
                                print('Epoch: ' + str(epoch) + 
                                        '\nError: ' + str(_cross_entropy) +
                                        '\nAccuracy: ' + str(_accuracy) + '\n')

                        else:
                            _, _error = sess.run([optimizer, error], feed_dict={X:_train_x, Y:_train_y })
                            
                            if epoch%10 == 0 and epoch > 0:
                                print('Epoch ' +  str((epoch+1)) + ' Error: ' + str(_error))
                                                               
                    
            test_error = []
            for _test_x, _test_y, in zip(test_x, test_y):
                test_error.append(sess.run(error, feed_dict={X:_test_x, Y:_test_y}))
            print('Test Error: ' + str(np.sum(test_error)))


class MLPModelKeras():
    
    def __init__(self, n_units, n_layers, n_columns, n_outputs, learning_rate, hidden_activation, output_activation):
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_outputs = n_outputs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate

    def create_policy_model(self, input_shape):
        input_layer = layers.Input(shape=input_shape)
        advantages = layers.Input(shape=[1])
        
        hidden_layer = layers.Dense(units=self.n_units, 
                                    activation=self.hidden_activation)(input_layer)
        
        output_layer = layers.Dense(units=self.n_outputs, 
                                    activation=self.output_activation)(hidden_layer)
        
        def log_likelihood_loss(y_true, y_pred):
            log_lik = backend.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
            return backend.mean(log_lik * advantages, keepdims=True)
        
        policy_model = Model(inputs=[input_layer, advantages], outputs=output_layer)
        policy_model.compile(loss=log_likelihood_loss, optimizer=Adam(self.learning_rate))
        model_prediction = Model(input=[input_layer], outputs=output_layer)
        return policy_model, model_prediction



        