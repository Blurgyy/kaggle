#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "zgy";

import torch
from torch import nn
import pickle
import random
from torch.autograd import Variable


def read_img():
    fn_name = "read_img";
    try:
        training_set = None;
        training_set_dmp_fname = "../dmp/training_set.pickle";
        with open(training_set_dmp_fname, 'rb') as f:
            training_set = pickle.load(f);
            random.shuffle(training_set);
        print("training set loaded and shuffled");
        return training_set;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

w = torch.zeros(10, 784);
def softmax_loss():
    fn_name = "softmax_loss";
    try:
        pass;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    w = Variable(torch.randn(10, 784), requires_grad = True);
    x_in = Variable(torch.randn(784, 1));
    x_b = Variable(torch.ones(784, 1), requires_grad = True);
    x = x_b * x_in;
    score = w.mm(x);
    score.backward(torch.ones(10, 1));
    print(w.grad);

if __name__ == '__main__':
    main();
