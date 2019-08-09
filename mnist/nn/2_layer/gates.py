#!/usr/bin/python3
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

