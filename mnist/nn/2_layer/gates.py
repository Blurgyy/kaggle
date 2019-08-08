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
    def forward(self, x, ):
        self.mask = (x > 0.01);
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
        self.z = np.dot(w, x);
        return self.z;
    def backward(self, dz, ):
        # @return dW and dx;
        self.dw = np.dot(dz, self.x.T);
        self.dx = np.dot(self.w.T, dz);
        # print(dz);
        # print(self.dw.shape, self.dx.shape);
        return [self.dw, self.dx];

