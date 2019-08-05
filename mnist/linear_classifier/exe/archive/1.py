#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import os 
import sys
import pickle 
import numpy as np 
from PIL import Image 
import random 
# import torch
# from torch.autograd import Variable 

training_set_dmp_fname = "../dmp/training_set.pickle";
# testing_set_dmp_fname = "../dmp/testing_set.pickle";
training_imgs = [];
# testing_img = [];
w_dmp_fname = "../dmp/w.pickle";

with open(training_set_dmp_fname, 'rb') as f:
    training_imgs = pickle.load(f);
    random.shuffle(training_imgs);
print("training set loaded and shuffled");
retrain = False;
if(len(sys.argv) > 1 and sys.argv[1] == "--retrain"):
    retrain = True;
if(os.path.exists(w_dmp_fname) and not retrain):
    try:
        with open(w_dmp_fname, 'rb') as f:
            w = pickle.load(f);
    except Exception as e:
        print("read from file failed: %s" % e);
        w = np.zeros(7840).reshape(10, 784);
        print("weight reinitialized");
else:
    w = np.zeros(7840).reshape(10, 784);

def regularization_loss(w, ):
    try:
        rows, cols = w.shape;
        ret = 0;
        for i in range(rows):
            for j in range(cols):
                ret += w[i][j] ** 2;
        return ret;
    except Exception as e:
        print("regularization_loss(): %s" % e);
        return None;

def softmax_loss(score, gt, ):
    try:
        ret = 0;
        smax = score;
        rows, cols = smax.shape;
        sum = 0;
        for i in range(rows):
            smax[i][0] = np.exp(score[i][0]);
            sum += smax[i][0];
        ret = -np.log(smax[gt][0]/sum);
        return ret;
    except Exception as e:
        print("softmax_loss(): %s" % e);
        return None;

def total_loss(w, score, ground_truth, Lambda, ):
    try:
        sloss = softmax_loss(score, ground_truth);
        rloss = regularization_loss(w);
        # print(sloss, rloss);
        return sloss + Lambda * rloss;
    except Exception as e:
        print("total_loss(): %s" % e);
        return None;

def update(w, x, loss, learning_rate, ):
    try:
        grad = x[0];
        print(grad);
        ground_truth = x[1];
        rows, cols = w.shape;
        for i in range(rows):
            for j in range(cols):
                # w[i][j] += learning_rate * loss * (-x[j][0]);
                if(i == ground_truth):
                    w[i][j] += learning_rate * loss * (grad[j][0]);
                else:
                    w[i][j] += learning_rate * loss * (-grad[j][0]);
        with open(w_dmp_fname, 'wb') as f:
            pickle.dump(w, f);
        # print(np.amax(w), np.amin(w), end = "\r");
        return w;
    except Exception as e:
        print("update(): %s" % e);
        return None;

def train(training_imgs, w, Lambda, learning_rate, ):
    try:
        YES = 0;
        NO = 0;
        # random.shuffle(training_imgs);
        size = len(training_imgs);
        for i in range(size):
            score = np.dot(w, training_imgs[i][0]);
            loss = total_loss(w, score, training_imgs[i][1], Lambda);
            
            maxx = -1;
            idx = -1;
            for j in range(score.shape[0]):
                if(maxx < score[j][0]):
                    maxx = score[j][0];
                    idx = j;
            # print("predicted:\t%d\nground truth:\t%d" % (idx, training_imgs[i][1]));
            if(idx == training_imgs[i][1]):
                YES += 1;
            else:
                NO += 1;
            # print(YES, NO);
            ratio = 100 * YES/(YES+NO);
            print("trained %d(%d/%d) pics, precision: %.2f%%" % (YES + NO, YES, NO, ratio), end = '\r');
            # print("trained %d(%d/%d) pics, precision: %.2f%%, (%g, %g)" % (YES + NO, YES, NO, ratio, np.amax(score), np.amin(score)), end = '\r');
            update(w, training_imgs[i], loss, learning_rate);
            # for i in range(score.shape[0]):
            #     print(score[i][0]);
            # input();
        return w;
    except Exception as e:
        print("train(): %s" % e);


Lambda = 0.02;
learning_rate = 0.000000005;
train(training_imgs, w, Lambda, learning_rate);

