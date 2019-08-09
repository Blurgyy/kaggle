#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import numpy as np
import gates 

def decay_schedule(length, name):
    i = np.arange(0, length, 1);
    if(name == "sigmoid"):
        return 1 / (1 + np.exp(i + 1 - length / 2));
    elif(name == "linear"):
        return (length - i) / length;
    elif(name == "hyperbola"):
        return 1 / (i + 1);
    elif(name == "constant"):
        return np.ones(length);

def init_model(input_size, hidden_layer_size, output_size, reg_strength):
    model = {};
    model['reg_strength'] = reg_strength;
    model['input'] = None;
    model['w1'] = 0.01 * np.random.randn(hidden_layer_size, input_size);
    model['w1.m'] = np.zeros(model['w1'].shape); # Adam update
    model['w1.v'] = model['w1.m']; # Adam update
    model['layer_1'] = gates.multiply_gate();
    model['h1'] = None;
    model['ReLU'] = gates.ReLU();
    model['w2'] = 0.01 * np.random.randn(output_size, hidden_layer_size);
    model['w2.m'] = np.zeros(model['w2'].shape); # Adam update
    model['w2.v'] = model['w2.m']; # Adam update
    model['layer_2'] = gates.multiply_gate();
    model['score'] = None;
    return model;

def forward(model, img, is_test_time):
    model['input'] = img;
    model['h1'] = model['layer_1'].forward(model['w1'], model['input']);
    model['h1'] = model['ReLU'].forward(model['h1'], is_test_time);
    model['score'] = model['layer_2'].forward(model['w2'], model['h1']);

def sgd_backward(model, dz, learning_rate):
    model['layer_2'].backward(dz);
    model['w2'] -= learning_rate * model['layer_2'].dw; # gradient descent 
    model['w2'] -= model['reg_strength'] * model['w2']; # regularization 
    model['ReLU'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU'].dx);
    model['w1'] -= learning_rate * model['layer_1'].dw; # gradient descent
    model['w1'] -= model['reg_strength'] * model['w1']; # regularization

def adam_backward(model, dz, iter_cnt, learning_rate):
    model['layer_2'].backward(dz);
    adam_update(model['layer_2'].dw, model['w2'], model['w2.m'], model['w2.v'], 
                iter_cnt, learning_rate, model['reg_strength']);
    model['w2'] -= model['reg_strength'] * model['w2']; # regularization 
    model['ReLU'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU'].dx);
    adam_update(model['layer_1'].dw, model['w1'], model['w1.m'], model['w1.v'], 
                iter_cnt, learning_rate, model['reg_strength']);
    model['w1'] -= model['reg_strength'] * model['w1']; # regularization

beta1 = 0.9;
beta2 = 0.995;
eps = 1e-7; # avoid deviding by zero
def adam_update(dx, x, m, v, iter_cnt, learning_rate, reg_strength):
    m = beta1 * m + (1 - beta1) * dx;
    v = beta2 * v + (1 - beta2) * (dx**2);
    m /= 1 - beta1**iter_cnt;
    v /= 1 - beta2**iter_cnt;
    x += -reg_strength * x; # regularization (first do regularization update then do Adam update)
    x += -learning_rate * m / (np.sqrt(v) + eps);
