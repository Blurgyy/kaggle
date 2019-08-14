#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import numpy as np

def im2col(batch, f_size, padding, stride, ):
    # batch: n * d * h * w
    # stride = stride;
    # f_size = f_size;
    n, d, h, w = batch.shape;
    p = padding;
    batch_pad = np.pad(batch, ((0,0), (0,0), (p,p), (p,p)));
    ret = [];
    for x in batch_pad:
        elem = None;
        for ver in range(0, h+p*2-f_size+1, stride):
            for hor in range(0, w+p*2-f_size+1, stride):
                # print(x[:, ver:ver+f_size, hor:hor+f_size]);
                # print(x[:, ver:ver+f_size, hor:hor+f_size].reshape(f_size*f_size*d, 1));
                # input();
                column = np.reshape(x[:, ver:ver+f_size, hor:hor+f_size], (f_size*f_size*d, 1));
                if(type(elem).__name__ == "NoneType"):
                    elem = column;
                else:
                    elem = np.hstack((elem, column));
        ret.append(elem);
    ret = np.array(ret);
    # print("im2col batch.shape", batch.shape)
    # print("im2col:\n", ret, ret.shape)
    return ret;

def col2im(batch, h, w):
    n, k, h_w = batch.shape;
    ret = [];
    for x in batch:
        elem = [];
        for i in range(k):
            elem.append(np.array(x[i, :]));
        ret.append(np.array(elem).reshape(k, h, w));
    ret = np.array(ret).reshape(n, k, h, w);
    print("col2im batch.shape:", batch.shape);
    print("col2im", ret.shape)
    return ret;

def fltr(size, depth):
    return 0.01 * np.random.randn(depth*size*size, 1);

class conv_layer:
    def __init__(self, k_filters, f_size, f_depth, stride, padding):
        # k: number of filters
        # f: filters' spatial extent
        # stride = 1
        # padding = 1
        self.k_filters = k_filters;
        self.f_size = f_size;
        self.f_depth = f_depth;
        self.stride = stride;
        self.padding = padding;
        self.filters = [];
        # self.bias = 
        self.init_filters();
    def init_filters(self, ):
        for i in range(self.k_filters):
            self.filters.append(fltr(size = self.f_size, depth = self.f_depth));
        self.filters = np.array(self.filters).reshape(self.k_filters, self.f_size*self.f_size*self.f_depth);
        self.df = np.zeros(self.filters.shape);
    def forward(self, x, ):
        # x: n * d * h * w
        self.x = x;
        n, d, h, w = self.x.shape;
        output_h = int((h + 2*self.padding - self.f_size) / self.stride + 1);
        output_w = int((w + 2*self.padding - self.f_size) / self.stride + 1);
        # x_reshaped = self.x.reshape(n*d, 1, h, w);
        self.x_col = im2col(self.x, f_size = self.f_size, padding = self.padding, stride = self.stride);
        self.z_col = [];
        for b in range(n):
            self.z_col.append(self.filters @ self.x_col[b]);
        self.z_col = np.array(self.z_col);
        self.z = col2im(self.z_col, output_h, output_w);
        return self.z;
    def backward(self, dz, ):
        # dz: n * k * h * w
        self.dz = dz;
        n, k, h, w = self.dz.shape;
        output_h = (h-1) * self.stride + self.f_size - 2*self.padding;
        output_w = (w-1) * self.stride + self.f_size - 2*self.padding;
        dz_reshaped = self.dz.reshape(n*k, 1, h, w);
        self.dz_col = im2col(dz_reshaped, f_size = self.f_size, padding = self.padding, stride = self.stride);
        self.dx = np.zeros(self.x.shape);
        self.dx_col = im2col(self.dx, f_size = self.f_size, padding = self.padding, stride = self.stride);
        """ caclulate df """
        for l in range(n*k):
            for i in range(h*w):
                self.df[l%k,:] += (self.dz_col[l,:,i] * self.x_col[int(l/k),:,i]);
        """ caclulate dx """
        print("self.x.shape", self.x.shape)
        print("self.dx_col.shape", self.dx_col.shape)
        print("self.dz_col.shape", self.dz_col.shape)
        print("self.filters.shape", self.filters.shape)
        for l in range(n*k):
            for i in range(h*w):
                self.dx_col[int(l/k),:,i] += self.dz_col[l,:,i] * self.filters[l%k,:];
                pass;
        self.dx = col2im(self.dx_col, output_h, output_w);

        # for b in range(n):
        #     for l in range(k):
        #         for i in range(h):
        #             for j in range(w):
        #                 self.dx[b, ] += self.dz[b][l][i][j] 
        # self.dx = np.dot(self.filters.reshape(1,-1), self.dz_col).reshape(self.x.shape);

        # self.df = []
        # for i in range(n):
        #     print(self.dz[i].reshape(self.k_filters, h*w));
        #     print(self.x_col[i].T);
        #     self.df.append(np.dot(self.dz[i].reshape(self.k_filters, h*w), self.x_col[i].T));
        # self.df = np.array(self.df);
        # # print(self.filters.shape)
        # # print(self.df.shape)
        # self.db = dz;
        return self.dx;
    def update(self, learning_rate, ):
        self.filters += -learning_rate * self.df;
        self.df = 0;

class pooling_layer:
    def __init__(self, size, stride, ):
        self.size = size;
        self.stride = stride;
        self.max_idx = [];
    def forward(self, x, ):
        self.z = [];
        d, h, w = x.shape;


        return self.z;


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
        self.z = self.w @ self.x + self.b;
        return self.z;
    def backward(self, dz, ):
        self.dw += dz @ self.x.T;
        self.db += dz;
        self.dx = self.w.T @ dz;
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
                                f_size = 3, f_depth = 1);
    model['relu1'] = ReLU();
    # model['pooling1'] = 
    model['conv2'] = conv_layer(k_filters = 16,
                                f_size = 3, f_depth = 4);
    model['relu2'] = ReLU();
    # model['pooling2'] = 
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
