#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

"""
functions:
    get_im2col_indices 
    im2col 
    col2im 
    decay_schedule 
    sample_batches 
    init_model 
    forward 
    backward 
    grad 
    update 

classes:
    conv_layer 
    pooling_layer 
    fc_layer 
    ReLU 
"""

import numpy as np

def get_im2col_indices(x_shape, filter_h, filter_w, padding, stride):
    N, C, H, W = x_shape
    out_h = int((H + 2 * padding - filter_h) / stride + 1)
    out_w = int((W + 2 * padding - filter_h) / stride + 1)

    i0 = np.repeat(np.arange(filter_h), filter_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)

    j0 = np.tile(np.arange(filter_w), filter_h * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    k = np.repeat(np.arange(C), filter_h * filter_w).reshape(-1, 1);
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (k, i, j)


def im2col(x, filter_h, filter_w, padding, stride):
    N, C, H, W = x.shape
    k, i, j = get_im2col_indices(x.shape, filter_h, filter_w, padding, stride)
    p = padding
    x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)))
    cols = x_pad[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(filter_h*filter_w*C, -1)
    return cols

def col2im(cols, x_shape, filter_h, filter_w, padding, stride):
    N, C, H, W = x_shape
    p = padding
    H_pad, W_pad = H + 2 * p, W + 2 * p
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype = cols.dtype)
    k, i, j = get_im2col_indices(x_shape, filter_h, filter_w, p, stride)
    cols_reshaped = cols.reshape(C * filter_h * filter_w, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_pad, (slice(None), k, i, j), cols_reshaped)
    if(p == 0):
        return x_pad
    return x_pad[:, :, p:-p, p:-p]

def fltr(size, depth):
    return 0.01 * np.random.randn(depth*size*size, 1);

class conv_layer:
    def __init__(self, k_filters, f_size, f_depth, padding, stride):
        # k: number of filters
        # f: filters' spatial extent
        # stride = 1
        # padding = 1
        self.k_filters = k_filters;
        self.f_size = f_size;
        self.f_depth = f_depth;
        self.stride = stride;
        self.padding = padding;
        self.init_filters();
        self.init_bias();
    def init_filters(self, ):
        self.filters = np.random.randn(self.k_filters, self.f_depth*self.f_size*self.f_size) / np.sqrt(self.f_size*self.f_size*self.f_depth);
    def init_bias(self, ):
        self.b = 0.01 * np.random.randn(self.k_filters, 1);
    def forward(self, x, ):
        self.x = x;
        N, C, H, W = self.x.shape
        self.x_col = im2col(self.x, filter_h = self.f_size, filter_w = self.f_size, 
                          padding = self.padding, stride = self.stride);
        # self.x_col.shape: (fh*fw*fd, positions)
        # self.filters.shape: (k_filters, fh*fw*d)
        self.z = self.filters @ self.x_col + self.b;
        # self.z.shape: (k_filters, positions)
        out_h = int((H + 2 * self.padding - self.f_size) / self.stride + 1);
        out_w = int((W + 2 * self.padding - self.f_size) / self.stride + 1);
        self.z = self.z.reshape(self.k_filters, out_h, out_w, N); # ...
        self.z = self.z.transpose(3, 0, 1, 2);
        # self.z.shape: (N, k_filters, out_h, out_w)
        return self.z;
    def backward(self, dz, ):
        # dz.shape: (N, k_filters, out_h, out_w)
        self.db = np.sum(dz, axis=(0, 2, 3));
        # self.db.shape: (self.k_filters, 1);
        dz_reshaped = dz.transpose(1, 2, 3, 0).reshape(self.k_filters, -1);
        self.df = dz_reshaped @ self.x_col.T;
        self.df = self.df.reshape(self.filters.shape);
        self.dx_col = self.filters.T @ dz_reshaped;
        self.dx = col2im(self.dx_col, self.x.shape, 
                         filter_h = self.f_size, filter_w = self.f_size, 
                         padding = self.padding, stride = self.stride);
        return self.dx;
    def update(self, learning_rate, ):
        assert self.b.size == self.db.size
        self.b += -learning_rate * self.db.reshape(self.k_filters, 1);
        self.filters += -learning_rate * self.df;

class pooling_layer:
    def __init__(self, size, padding, stride, ):
        self.size = size;
        self.padding = padding;
        self.stride = stride;
    def forward(self, x, ):
        self.x = x;
        N, C, H, W = self.x.shape;
        out_h = int((H + 2 * self.padding - self.size) / self.stride + 1);
        out_w = int((W + 2 * self.padding - self.size) / self.stride + 1);
        x_reshaped = x.reshape(N*C, 1, H, W);
        self.x_col = im2col(x_reshaped, filter_h = self.size, filter_w = self.size, 
                                   padding = self.padding, stride = self.stride);
        # self.x_col.shape: (fh*hw*d, positions)
        self.max_idx = np.argmax(self.x_col, axis=0);
        self.z = self.x_col[self.max_idx, range(self.max_idx.size)];
        self.z = self.z.reshape(out_h, out_w, N, C);
        self.z = self.z.transpose(2, 3, 0, 1)
        return self.z;
    def backward(self, dz, ):
        N, C, H, W = self.x.shape
        dx_col = np.zeros_like(self.x_col);
        dz = dz.transpose(2, 3, 0, 1); # ...
        dz_flat = dz.ravel();
        dx_col[self.max_idx, range(self.max_idx.size)] = dz_flat;
        self.dx = col2im(dx_col, (N*C, 1, H, W), 
                         filter_h = self.size, filter_w = self.size, 
                         padding = self.padding, stride = self.stride);
        self.dx = self.dx.reshape(self.x.shape);
        return self.dx;

class bn_layer:
    def __init__(self, ):
        pass;
    def forward(self, x, ):
        pass;
    def backward(self, dz, ):
        pass;
    def update(self, learning_rate, ):
        pass;

class fc_layer:
    def __init__(self, input_size, output_size, reg, ):
        self.input_size = input_size;
        self.output_size = output_size;
        self.reg = reg;
        self.init_weights();
        self.init_bias();
    def init_weights(self, ):
        self.w = np.random.randn(self.output_size, self.input_size) / np.sqrt(self.input_size/2);
    def init_bias(self):
        self.b = np.random.randn(self.output_size, 1) / np.sqrt(self.input_size);
    def forward(self, x, ):
        self.x = x;
        N = self.x.shape[0];
        x_reshaped = self.x.transpose(2, 1, 0).reshape(self.input_size, N);
        try:
            self.z = self.w @ x_reshaped + self.b;
        except RuntimeWarning:
            print(np.max(self.w));
            print(np.max(x_reshaped));
            exit();
        self.z = self.z.T.reshape(N, self.output_size, 1);
        return self.z;
    def backward(self, dz, ):
        # print("\nfc_layer::backward()")
        N = self.x.shape[0];
        self.db = np.sum(dz, axis=(0, 2));
        self.db = self.db.reshape(self.output_size, -1)
        dz_reshaped = dz.transpose(2, 1, 0).reshape(self.output_size, N);
        x_reshaped = self.x.transpose(2, 0, 1).reshape(N, self.input_size);
        self.dw = dz_reshaped @ x_reshaped;
        self.dx = self.w.T @ dz_reshaped;
        self.dx = self.dx.T.reshape(N, self.input_size, 1);
        return self.dx;
    def update(self, learning_rate, ):
        self.w += -self.reg * learning_rate * self.w;
        self.w += -learning_rate * self.dw;
        self.b += -learning_rate * self.db;

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
        return 0.8 ** (i);

def init_model(reg):
    model = {};
    model['conv1'] = conv_layer(k_filters = 4,
                                f_size = 3, f_depth = 1,
                                padding = 1, stride = 1);
    model['relu1'] = ReLU();
    model['pooling1'] = pooling_layer(size = 2, padding = 0, stride = 2);
    model['conv2'] = conv_layer(k_filters = 16,
                                f_size = 3, f_depth = 4,
                                padding = 1, stride = 1);
    model['relu2'] = ReLU();
    model['pooling2'] = pooling_layer(size = 2, padding = 0, stride = 2);
    model['fc6'] = fc_layer(input_size = 784, output_size = 200, reg = reg);
    model['relu3'] = ReLU();
    model['fc7'] = fc_layer(input_size = 200, output_size = 10, reg = reg);
    model['output'] = None;
    return model;

def forward(model, x, is_test_time):
    x = model['conv1'].forward(x);
    x = model['relu1'].forward(x, is_test_time);
    x = model['pooling1'].forward(x);
    x = model['conv2'].forward(x);
    x = model['relu2'].forward(x, is_test_time);
    x = model['pooling2'].forward(x);
    N, C, H, W = x.shape;
    x = x.reshape(N, -1, 1);
    x = model['fc6'].forward(x);
    x = model['relu3'].forward(x, is_test_time);
    model['output'] = model['fc7'].forward(x);
    return model['output'];

def backward(model, dz):
    dz = model['fc7'].backward(dz);
    dz = model['relu3'].backward(dz);
    dz = model['fc6'].backward(dz);

    # N, C, H, W = model['pooling2'].z.shape;
    # dz = dz.reshape(N, C, H, W);
    dz = dz.reshape(model['pooling2'].z.shape);
    dz = model['pooling2'].backward(dz);
    dz = model['relu2'].backward(dz);
    dz = model['conv2'].backward(dz);
    dz = model['pooling1'].backward(dz);
    dz = model['relu1'].backward(dz);
    model['conv1'].backward(dz);

def grad(model, y):
    prob = model['output'].copy();
    prob -= np.max(prob, axis=1).reshape(-1,1,1);
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1).reshape(-1,1,1);
    # print("check:\n", prob, y);
    dz = prob.copy();
    a0 = np.arange(len(y)).reshape(-1,1);
    y = y.reshape(-1,1);
    a2 = np.repeat(0, len(y)).reshape(-1,1);
    # dz[a0,y,a2] -= 1;
    np.add.at(dz, (a0,y,a2), -1);
    try:
        loss = np.sum(-np.log(prob[a0,y,a2]))
    except RuntimeWarning:
        print("\nwhat's in log():");
        print(prob[a0,y,a2]);
        exit();
    return dz, loss;

def update(model, learning_rate):
    model['fc7'].update(learning_rate);
    model['fc6'].update(learning_rate);
    # model['pooling2'].update(learning_rate);
    model['conv2'].update(learning_rate);
    # model['pooling1'].update(learning_rate);
    model['conv1'].update(learning_rate);
