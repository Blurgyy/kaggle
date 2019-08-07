#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import numpy as np


def descend_pattern(length, name):
    i = np.arange(0, length, 1);
    if(name == "sigmoid"):
        return 1 / (1 + np.exp(i + 1 - length / 2));
    elif(name == "linear"):
        return (length - i) / length;
    elif(name == "hyperbola"):
        return 1 / (i + 1);

def forward(model, img):
    model['input'] = img;
    model['h1'] = model['layer_1'].forward(model['w1'], model['input']);
    model['h1'] = model['ReLU'].forward(model['h1']);
    model['score'] = model['layer_2'].forward(model['w2'], model['h1']);

def backward(model, dz, learning_rate):
    model['layer_2'].backward(dz);
    model['w2'] -= learning_rate * model['layer_2'].dw; # gradient descent
    model['w2'] -= 1e-3 * learning_rate * model['w2']; # regularization 
    model['ReLU'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU'].dx);
    model['w1'] -= learning_rate * model['layer_1'].dw; # gradient descent
    model['w1'] -= 1e-3 * learning_rate * model['w1']; # regularization
