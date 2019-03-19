#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 06:54:29 2019

@author: tawehbeysolow
"""

import tensorflow as tf, numpy as np
from baselines.a2c.utils import mse
from baselines.common.runners import AbstractEnvRunner

def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()

class ActorCriticModel(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self,
                 policy,
                ob_space,
                action_space,
                nenvs,
                nsteps,
                ent_coef,
                vf_coef,
                max_grad_norm):

        sess = tf.get_default_session()

        # Here we create the placeholders
        actions_ = tf.placeholder(tf.int32, [None], name="actions_")
        advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
        rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
        lr_ = tf.placeholder(tf.float32, name="learning_rate_")

        # Here we create our two models:
        # Step_model that is used for sampling
        step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)

        # Train model for training
        train_model = policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True)

        """
        Calculate the loss
        Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        """
        # Policy loss
        # Output -log(pi)
        negative_log_pi = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)

        # 1/n * sum A(si,ai) * -logpi(ai|si)
        pg_loss = tf.reduce_mean(advantages_ * negative_log_pi)

        # Value loss 1/2 SUM [R - V(s)]^2
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf),rewards_))

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())


        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("model")

        # 2. Calculate the gradients
        gradients = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
            
        gradients = list(zip(gradients, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Build our trainer
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr_, decay=0.99, epsilon=1e-5)

        # 4. Backpropagation
        _train = trainer.apply_gradients(gradients)
        

        def train(states_in, actions, returns, values, lr):
            advantages = returns - values

            # We create the feed dictionary
            td_map = {train_model.inputs_: states_in,
                     actions_: actions,
                     advantages_: advantages, # Use to calculate our policy loss
                     rewards_: returns, # Use as a bootstrap for real value
                     lr_: lr}

            policy_loss, value_loss, policy_entropy, _= sess.run([pg_loss, vf_loss, entropy, _train], td_map)
            
            return policy_loss, value_loss, policy_entropy


        def save(save_path):
            """
            Save the model
            """
            saver = tf.train.Saver()
            saver.save(sess, save_path)

        def load(load_path):
            """
            Load the model
            """
            saver = tf.train.Saver()
            print('Loading ' + load_path)
            saver.restore(sess, load_path)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    
    __init__:
    - Initialize the runner
    run():
        
    - Make a mini batch
    """
    def __init__(self, environment, model, nsteps, total_timesteps, gamma, _lambda):
        super().__init__(environment = environment, model = model, nsteps = nsteps)

        # Discount rate
        self.gamma = gamma

        # Lambda used in GAE (General Advantage Estimation)
        self._lambda = _lambda

        # Total timesteps taken
        self.total_timesteps = total_timesteps

    def run(self):
        # Here, we init the lists that will contain the batch of experiences
        _observations, _actions, _rewards, _values, _dones = [],[],[],[],[]

        # For n in range number of steps
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because AbstractEnvRunner run self.obs[:] = env.reset()
            actions, values = self.model.step(self.obs, self.dones)
            _observations.append(np.copy(self.obs)) #obs len nenvs (1 step per env)
            _actions.append(actions)
            _values.append(values)
            _dones.append(self.dones)
            self.observations[:], rewards, self.dones, _ = self.environment.step(actions)
            _rewards.append(rewards)

        #batch of steps to batch of rollouts
        _observations = np.asarray(_observations, dtype=np.uint8)
        _rewards = np.asarray(_rewards, dtype=np.float32)
        _actions = np.asarray(_actions, dtype=np.int32)
        _values = np.asarray(_values, dtype=np.float32)
        _dones = np.asarray(_dones, dtype=np.bool)
        last_values = self.model.value(self.observations)
        _returns = np.zeros_like(_rewards)
        _advantages = np.zeros_like(_rewards)

        last_lambda = 0

        # From last step to first step
        for t in reversed(range(self.nsteps)):
            # If t == before last step
            if t == self.nsteps - 1:
                # If a state is done, nextnonterminal = 0
                # In fact nextnonterminal allows us to do that logic

                #if done (so nextnonterminal = 0):
                #    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0) 
                # else (not done)
                    #delta = R + gamma * V(st+1)
                nextnonterminal = 1.0 - self.dones
                
                # V(t+1)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - _dones[t+1]
                
                nextvalues = _values[t+1]

            delta = _rewards[t] + self.gamma * nextvalues * nextnonterminal - _values[t]
            _advantages[t] = last_lambda = delta + self.gamma * self._lambda * nextnonterminal * last_lambda

        # Returns
        _returns = _advantages + _values

        return map(swap_flatten_axes, (_observations, _actions, _returns, _values))


def swap_flatten_axes(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
