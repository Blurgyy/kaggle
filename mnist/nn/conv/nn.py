#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import numpy as np

def conv(x, f, s = 1):
    # x: input;
    # f: filter
    # s: stride
    # requires x and f to have same depth
    depth, f_wid, f_hei = f.shape;
    ret = [];
    row = [];
    for ver in range(0, x.shape[1]-f_hei+1, s):
        for hor in range(0, x.shape[2]-f_wid+1, s):
            row.append(np.sum(x[:, ver:ver+f_wid, hor:hor+f_hei] * f));
        ret.append(row);
        row = [];
    ret = np.array(ret);
    return ret;

# class fltr:
#     def __init__(self, size, depth, ):
#         self.size = size;
#         self.depth = depth;
#         self.init_weights();
#     def init_weights(self, ):
#         self.f = 0.001 * np.random.randn(self.depth, self.size, self.size);

def fltr(size, depth):
    return 0.01 * np.random.randn(depth, size, size);

class conv_layer:
    def __init__(self, k_filters, f_size, f_depth, stride, padding, ):
        # k: number of filters
        # f: filters' spatial extent
        # stride: stride
        # padding: amount of zero padding
        self.k_filters = k_filters;
        self.f_size = f_size;
        self.f_depth = f_depth;
        self.stride = stride;
        self.padding = padding;
        self.filters = [];
        # self.bias = 
        self.init_filters();
        self.df = [];
    def init_filters(self, ):
        for i in range(self.k_filters):
            self.filters.append(fltr(size = self.f_size, depth = self.f_depth));
    def forward(self, x, ):
        # x: w1 * h1 * d1
        # do padding in advance
        self.x = np.pad(x, ((0,0), (self.padding, self.padding), (self.padding, self.padding)));
        self.z = [];
        for f in self.filters:
            elem = conv(self.x, f, self.stride)
            self.z.append(elem);
        self.z = np.array(self.z);
        return self.z;
    def backward(self, dz, ):
        # print(dz.shape);
        if(self.df == []):
            for i in range(0, self.k_filters):
                self.df.append(np.zeros((self.f_depth, self.f_size, self.f_size)));
        self.dx = np.zeros(self.x.shape);
        for i in range(0, dz.shape[0]):
            # for f in self.filters:
            for j in range(self.k_filters):
                # print(self.dx.shape);
                for ver in range(0, self.dx.shape[1]-self.f_size+1, self.stride):
                    for hor in range(0, self.dx.shape[2]-self.f_size+1, self.stride):
                        self.dx[:, ver:ver+self.f_size, hor:hor+self.f_size] += self.filters[j] * dz[i,int(ver/self.stride),int(hor/self.stride)];
                        self.df[j] += self.x[:, ver:ver+self.f_size, hor:hor+self.f_size] * dz[i,int(ver/self.stride),int(hor/self.stride)];
        self.dx = self.dx[:, self.padding:self.dx.shape[1]-self.padding, self.padding:self.dx.shape[2]-self.padding];
        self.db = dz;
        return self.dx;
    def update(self, learning_rate, ):
        for i in range(self.k_filters):
            self.filters[i] += -learning_rate * self.df[i];
        self.df = [];

class fc_layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size;
        self.output_size = output_size;
        self.init_weights();
        self.init_bias();

        self.dw = 0;
        self.dx = 0;
        self.db = 0;
    def init_weights(self, ):
        self.w = 0.01 * np.random.randn(self.output_size, self.input_size);
    def init_bias(self):
        self.b = 0.01 * np.random.randn(self.output_size, 1);
    def forward(self, x, ):
        self.x = x;
        # print(self.w.shape, self.x.shape);
        self.z = np.dot(self.w, self.x) + self.b;
        return self.z;
    def backward(self, dz, ):
        self.dw += np.dot(dz, self.x.T);
        self.db += dz;
        self.dx = np.dot(self.w.T, dz);
        return self.dx;
    def update(self, learning_rate, ):
        self.w += -learning_rate * self.dw;
        self.b += -learning_rate * 0.1 * self.db;
        self.dw = 0;
        self.db = 0;

class ReLU:
    def __init__(self, ):
        self.p = 0.5; # dropout ratio
    def forward(self, x, is_test_time):
        # if is_test_time == True, disable dropout
        self.relu_mask = (x > 0.01);
        if(is_test_time):
            self.dropout_mask = self.p;
        else:
            self.dropout_mask = (np.random.rand(*x.shape) < self.p);
        self.mask = self.relu_mask * self.dropout_mask;
        self.z = x * self.mask;
        return self.z;
    def backward(self, dz, ):
        self.dx = dz * self.mask;
        return self.dx;

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

def init_model():
    model = {};
    model['input'] = None;
    model['conv1'] = conv_layer(k_filters = 4,
                                f_size = 3, f_depth = 1,
                                stride = 1, padding = 1);
    model['relu1'] = ReLU();
    model['pooling1'] = conv_layer(k_filters = 4,
                                   f_size = 2, f_depth = 4,
                                   stride = 2, padding = 0);
    model['conv2'] = conv_layer(k_filters = 16,
                                f_size = 3, f_depth = 4,
                                stride = 1, padding = 1);
    model['relu2'] = ReLU();
    model['pooling2'] = conv_layer(k_filters = 16,
                                   f_size = 2, f_depth = 16,
                                   stride = 2, padding = 0);
    model['fc6'] = fc_layer(input_size = 784, output_size = 200);
    model['relu3'] = ReLU();
    model['fc7'] = fc_layer(input_size = 200, output_size = 10);
    model['output'] = None;
    return model;

def forward(model, x, is_test_time):
    model['input'] = x;
    x = model['conv1'].forward(x);
    x = model['relu1'].forward(x, is_test_time);
    x = model['pooling1'].forward(x);
    x = model['conv2'].forward(x);
    x = model['relu2'].forward(x, is_test_time);
    x = model['pooling2'].forward(x);
    x = x.reshape(784, 1);
    x = model['fc6'].forward(x);
    # print("fc6 output:", x.shape);
    x = model['relu3'].forward(x, is_test_time);
    model['output'] = model['fc7'].forward(x);
    # print("fc7 output:", model['output'].shape);
    return model['output'];

def backward(model, dz):
    model['fc7'].backward(dz);
    model['relu3'].backward(model['fc7'].dx);
    model['fc6'].backward(model['relu3'].dx);
    model['pooling2'].backward(model['fc6'].dx.reshape(16, 7, 7));
    model['relu2'].backward(model['pooling2'].dx);
    model['conv2'].backward(model['relu2'].dx);
    model['pooling1'].backward(model['conv2'].dx);
    model['relu1'].backward(model['pooling1'].dx);
    model['conv1'].backward(model['relu1'].dx);

def update(model, learning_rate):
    model['fc7'].update(learning_rate);
    model['fc6'].update(learning_rate);
    model['pooling2'].update(learning_rate);
    model['conv2'].update(learning_rate);
    model['pooling1'].update(learning_rate);
    model['conv1'].update(learning_rate);
