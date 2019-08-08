#!/usr/bin/python3
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

def init_model(input_size, hidden_layer_size, output_size, reg_strength):
    model = {};
    model['reg_strength'] = reg_strength;
    model['input'] = None;
    model['w1'] = 0.01 * np.random.randn(hidden_layer_size, input_size);
    # model['w1'] = np.zeros(hidden_layer_size * input_size).reshape(hidden_layer_size, input_size);
    model['layer_1'] = gates.multiply_gate();
    model['h1'] = None;
    model['ReLU'] = gates.ReLU();
    model['w2'] = 0.01 * np.random.randn(output_size, hidden_layer_size);
    # model['w2'] = np.zeros(output_size * hidden_layer_size).reshape(output_size, hidden_layer_size);
    model['layer_2'] = gates.multiply_gate();
    model['score'] = None;
    return model;

def forward(model, img):
    model['input'] = img;
    model['h1'] = model['layer_1'].forward(model['w1'], model['input']);
    model['h1'] = model['ReLU'].forward(model['h1']);
    model['score'] = model['layer_2'].forward(model['w2'], model['h1']);

def backward(model, dz, learning_rate):
    model['layer_2'].backward(dz);
    model['w2'] -= learning_rate * model['layer_2'].dw; # gradient descent 
    model['w2'] -= model['reg_strength'] * model['w2']; # regularization 
    model['ReLU'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU'].dx);
    model['w1'] -= learning_rate * model['layer_1'].dw; # gradient descent
    model['w1'] -= model['reg_strength'] * model['w1']; # regularization
