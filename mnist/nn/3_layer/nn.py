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
    def forward(self, w, x, ):
        self.w = w;
        self.x = x;
        self.z = np.dot(self.w, self.x);
        return self.z;
    def backward(self, dz, ):
        # @return dW and dx;
        self.dw = np.dot(dz, self.x.T);
        self.dx = np.dot(self.w.T, dz);
        # print(dz);
        # print(self.dw.shape, self.dx.shape);
        return [self.dw, self.dx];


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
        return 0.9 ** (i);

def init_model(input_size, h1_size, h2_size, output_size, reg_strength):
    model = {};
    model['reg_strength'] = reg_strength;
    model['iter_cnt'] = 0;
    model['input'] = None;

    model['w1'] = 0.01 * np.random.randn(h1_size, input_size);
    model['w1.m'] = np.zeros(model['w1'].shape); # Adam update
    model['w1.v'] = np.zeros(model['w1'].shape); # Adam update
    model['layer_1'] = multiply_gate();
    model['h1'] = None;
    model['ReLU_1'] = ReLU();

    model['w2'] = 0.01 * np.random.randn(h2_size, h1_size);
    model['w2.m'] = np.zeros(model['w2'].shape); # Adam update
    model['w2.v'] = np.zeros(model['w2'].shape); # Adam update
    model['layer_2'] = multiply_gate();
    model['h2'] = None;
    model['ReLU_2'] = ReLU();

    model['w3'] = 0.01 * np.random.randn(output_size, h2_size);
    model['w3.m'] = np.zeros(model['w3'].shape); # Adam update
    model['w3.v'] = np.zeros(model['w3'].shape); # Adam update
    model['layer_3'] = multiply_gate();
    model['score'] = None;
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
    model['h1'] = model['layer_1'].forward(model['w1'], model['input']);
    model['h1'] = model['ReLU_1'].forward(model['h1'], is_test_time);
    model['h2'] = model['layer_2'].forward(model['w2'], model['h1']);
    model['h2'] = model['ReLU_2'].forward(model['h2'], is_test_time);
    model['score'] = model['layer_3'].forward(model['w3'], model['h2']);

def sgd_backward(model, dz, learning_rate):
    model['iter_cnt'] += 1;
    model['layer_3'].backward(dz);
    model['w3'] -= model['reg_strength'] * model['w3'];
    model['w3'] -= learning_rate * model['layer_3'].dw;
    model['ReLU_2'].backward(model['layer_3'].dx);
    model['layer_2'].backward(model['ReLU_2'].dx);
    model['w2'] -= model['reg_strength'] * model['w2'];
    model['w2'] -= learning_rate * model['layer_2'].dw;
    model['ReLU_1'].backward(model['layer_2'].dx);
    model['layer_1'].backward(model['ReLU_1'].dx);
    model['w1'] -= model['reg_strength'] * model['w1'];
    model['w1'] -= learning_rate * model['layer_1'].dw;

def adam_backward(model, dz, learning_rate):
    model['iter_cnt'] += 1;
    model['layer_3'].backward(dz);
    adam_update(model['layer_3'].dw, model['w3'], model['w3.m'], model['w3.v'], 
                model['iter_cnt'], learning_rate, model['reg_strength']);
    model['ReLU_2'].backward(model['layer_3'].dx);

    model['layer_2'].backward(model['ReLU_2'].dx);
    adam_update(model['layer_2'].dw, model['w2'], model['w2.m'], model['w2.v'], 
                model['iter_cnt'], learning_rate, model['reg_strength']);
    model['ReLU_1'].backward(model['layer_2'].dx);

    model['layer_1'].backward(model['ReLU_1'].dx);
    adam_update(model['layer_1'].dw, model['w1'], model['w1.m'], model['w1.v'],
                model['iter_cnt'], learning_rate, model['reg_strength']);
    

beta1 = 0.9;
beta2 = 0.995;
eps = 1e-7; # prevent devision by zero
def adam_update(dx, x, m, v, iter_cnt, learning_rate, reg_strength):
    m = beta1 * m + (1 - beta1) * dx;
    v = beta2 * v + (1 - beta2) * (dx**2);
    # m /= 1 - beta1**iter_cnt;
    # v /= 1 - beta2**iter_cnt;
    x += -reg_strength * x; # regularization (first do regularization update then do Adam update)
    x += -learning_rate * m / (np.sqrt(v) + eps);
