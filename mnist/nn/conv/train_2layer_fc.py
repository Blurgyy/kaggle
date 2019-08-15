#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import nn 
import plot 
import click 

import warnings
warnings.filterwarnings("error")

def fc_model(input_size, h1_size, output_size):
    model = {};
    model['fc1'] = nn.fc_layer(input_size, h1_size, 1e-3);
    model['relu'] = nn.ReLU();
    model['drop'] = nn.dropout_layer(0.5);
    model['fc2'] = nn.fc_layer(h1_size, output_size, 1e-3);
    model['output'] = None;
    return model;

def forward(model, x):
    x = model['fc1'].forward(x);
    x = model['relu'].forward(x);
    x = model['drop'].forward(x, False)
    model['output'] = model['fc2'].forward(x);

def backward(model, dz):
    dz = model['fc2'].backward(dz);
    dz = model['drop'].backward(dz);
    dz = model['relu'].backward(dz);
    model['fc1'].backward(dz);

def update(model, lr):
    model['fc1'].update(lr);
    model['fc2'].update(lr);

def main():
    epoch = 10000;
    lr = 1e-4;
    batch_size = 20;
    model = fc_model(784, 200, 10);

    train = data.preprocess_training_set();
    for ep in range(epoch):
        yes, cnt = 0, 0;
        X, Y = data.sample_batches_train(train, batch_size);
        for i in range(len(X)):
            x, y = X[i].reshape(batch_size, 784, 1), Y[i];
            forward(model, x);
            dz, loss = nn.grad(model, y);
            backward(model, dz);
            # print(np.max(dz), '   \t', np.min(dz))
            # print(np.max(model['fc1'].w), '\t', np.min(model['fc1'].w))
            # print(np.max(model['fc1'].b), '\t', np.min(model['fc1'].b))
            update(model, lr);

            prediction = np.argmax(model['output'], axis=1);
            score = prediction.reshape(-1,1) == y.reshape(-1,1)
            # print(score)
            yes += np.sum(score);
            cnt += batch_size;
            # print("???", prob[a0,y,a2])
            if(cnt % batch_size == 0):
                print("[%d/%d]: %.2f%%, loss = %.2f" % (yes, cnt, yes / cnt * 100, loss), end = '\r');
            # input()
        print()

if __name__ == '__main__':
    main()
