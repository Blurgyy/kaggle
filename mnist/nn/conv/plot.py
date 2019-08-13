#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import matplotlib.pyplot as plt 

class plot:
    def __init__(self, ):
        self.x = [];
        self.y = [];
        self.size = 0;
    def append(self, y_value, ):
        self.size += 1;
        self.x.append(self.size);
        self.y.append(y_value);
    def update(self, ):
        plt.figure();
        self.curve = plt.plot(self.x, self.y);
    def show(self, ):
        self.update();
        plt.show(self.curve);
        plt.close()
    def save(self, Path, ):
        self.update();
        plt.savefig(Path);
        plt.close();
