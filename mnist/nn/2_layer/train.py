#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import gates 
import pickle
import random

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

def main():
    input_size = 784;
    hidden_layer_size = 200;
    output_size = 10;
    model = data.init_model(input_size, hidden_layer_size, output_size);

    epoch = 20;
    i = np.arange(0, epoch, 1);
    base_learning_rate = 1e-5;
    # descend_pattern = 1 / (1 + np.exp(i + 1 - epoch / 2)); # sigmoid
    descend_pattern = (epoch - i) / epoch; # linear
    learning_rate = base_learning_rate * descend_pattern;
    # print(learning_rate);

    for ep in range(epoch):
        lr = learning_rate[ep];
        print("training epoch %d/%d with learning rate %g" % (ep+1, epoch, lr));
        training_set = data.preprocess_training_set();
        yes = 0;
        cnt = 0;
        for elem in training_set:
            label = elem[0];
            img = elem[1];
            forward(model, img);
            prob = model['score'];
            prob -= np.max(prob);
            prob = np.exp(prob) / np.sum(np.exp(prob));
            dz = prob;
            dz[label] -= 1;
            backward(model, dz, lr);

            predict = np.argmax(model['score']);
            yes += (predict == label);
            cnt += 1;
            if(cnt % 100 == 0):
                print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100), end = '\r');
        data.save_model(model);
        print("\nmodel saved");
        print();


if(__name__ == "__main__"):
    main();
