#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import numpy as np

class gate:
    def __init__(self, ):
        self.z = None;

class ReLU(gate):
    def __init__(self, ):
        super().__init__();
        self.p = 0.5; # dropout ratio
    def forward(self, x, is_test_time):
        self.relu_mask = (x > 0.01);
        if(is_test_time):
            self.dropout_mask = self.p;
        else:
            self.dropout_mask = (np.random.rand(*x.shape) < self.p);
        self.mask = self.relu_mask * self.dropout_mask;
        self.z = x * self.mask;
        # print(np.max(self.z));
        return self.z;
    def backward(self, dz):
        self.dx = dz * self.mask;
        return self.dx;

class multiply_gate(gate):
    def __init__(self, ):
        super().__init__();
    def forward(self, w, x, b, ):
        self.w = w;
        self.x = x;
        self.b = b;
        self.z = np.dot(self.w, self.x) + self.b;
        return self.z;
    def backward(self, dz, ):
        # @return dW and dx;
        self.dw = np.dot(dz, self.x.T);
        self.dx = np.dot(self.w.T, dz);
        self.db = dz;
        # print(dz);
        # print(self.dw.shape, self.dx.shape);
        return [self.dw, self.dx, self.db];

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
    elif(name == "exponential"):
        return 0.95 ** (i);

def init_model(input_size, hidden_layer_size, output_size, reg_strength):
    model = {};
    model['reg_strength'] = reg_strength;
    model['input'] = None;
    model['w1'] = 0.01 * np.random.randn(hidden_layer_size, input_size);
    model['w1.m'] = np.zeros(model['w1'].shape); # Adam update
    model['w1.v'] = np.zeros(model['w1'].shape); # Adam update
    model['layer_1'] = multiply_gate();
    model['h1'] = None;
    model['ReLU'] = ReLU();
    model['w2'] = 0.01 * np.random.randn(output_size, hidden_layer_size);
    model['w2.m'] = np.zeros(model['w2'].shape); # Adam update
    model['w2.v'] = np.zeros(model['w2'].shape); # Adam update
    model['layer_2'] = multiply_gate();
    model['output'] = None;
    model['iter_cnt'] = 0;
    model['w1.grad'] = np.zeros(model['w1'].shape);
    model['w2.grad'] = np.zeros(model['w2'].shape);

    model['b1'] = np.random.randn(hidden_layer_size, 1);
    model['b2'] = np.random.randn(output_size, 1);
    model['b1.grad'] = np.zeros(model['b1'].shape);
    model['b2.grad'] = np.zeros(model['b2'].shape);
    return model;

def sample_batches(training_set, batch_size):
    batch = [];
    ret = [];
    for elem in training_set:
        batch.append(elem);
        if(len(batch) == batch_size):
            ret.append(batch);
            batch = [];
    if(len(batch) > 0):
        ret.append(batch);
    return ret;

def forward(model, img, is_test_time):
    model['input'] = img;
    model['h1'] = model['layer_1'].forward(model['w1'], model['input'], model['b1']);
    model['h1'] = model['ReLU'].forward(model['h1'], is_test_time);
    model['score'] = model['layer_2'].forward(model['w2'], model['h1'], model['b2']);

def update_weights(model, learning_rate):
    model['w1'] += model['reg_strength'] * learning_rate * model['w1'];
    model['w2'] += model['reg_strength'] * learning_rate * model['w2'];
    model['w1'] += -learning_rate * model['w1.grad'];
    model['w2'] += -learning_rate * model['w2.grad'];
    model['w1.grad'] *= 0;
    model['w2.grad'] *= 0;

    model['b1'] += -learning_rate * model['b1.grad'];
    model['b2'] += -learning_rate * model['b2.grad'];
    model['b1.grad'] *= 0;
    model['b2.grad'] *= 0;

def sgd_backward(model, dz, batch_size):
    model['layer_2'].backward(dz);
    model['w2.grad'] += model['layer_2'].dw / batch_size;
    model['b2.grad'] += model['layer_2'].db / batch_size;
    model['ReLU'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU'].dx);
    model['w1.grad'] += model['layer_1'].dw / batch_size;
    model['b1.grad'] += model['layer_1'].db / batch_size;

def adam_backward(model, dz, learning_rate):
    model['iter_cnt'] += 1;
    model['layer_2'].backward(dz);
    adam_update(model['layer_2'].dw, model['w2'], model['w2.m'], model['w2.v'], 
                model['iter_cnt'], learning_rate, model['reg_strength']);
    # model['w2'] -= model['reg_strength'] * model['w2']; # regularization 
    model['ReLU'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU'].dx);
    adam_update(model['layer_1'].dw, model['w1'], model['w1.m'], model['w1.v'], 
                model['iter_cnt'], learning_rate, model['reg_strength']);
    # model['w1'] -= model['reg_strength'] * model['w1']; # regularization

beta1 = 0.9;
beta2 = 0.995;
eps = 1e-7; # prevent devision by zero
def adam_update(dx, x, m, v, iter_cnt, learning_rate, reg_strength):
    m = beta1 * m + (1 - beta1) * dx;
    v = beta2 * v + (1 - beta2) * (dx**2);
    # m /= 1 - beta1**iter_cnt;
    # v /= 1 - beta2**iter_cnt;
    x += -learning_rate * m / (np.sqrt(v) + eps);
