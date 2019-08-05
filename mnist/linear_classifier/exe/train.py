#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import os 
import sys
import pickle 
import numpy as np 
from PIL import Image 
import random 


def load_training_set():
    fn_name = "load_training_set";
    try:
        training_set_dmp_fname = "../dmp/training_set.pickle";
        training_imgs = None;
        with open(training_set_dmp_fname, 'rb') as f:
            training_imgs = pickle.load(f);
            random.shuffle(training_imgs);
        print("training set loaded and shuffled");
        return training_imgs;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def init_weights():
    fn_name = "init_weights";
    try:
        w_dmp_fname = "../dmp/w.pickle";
        if("--continue" in sys.argv):
            try:
                tmp_fname = sys.argv[sys.argv.index("--continue")+1];
                if(os.path.exists(tmp_fname)):
                    w_dmp_fname = tmp_fname;
                else:
                    print("file %s does not exist, ignoring" % tmp_fname);
            except Exception as e:
                pass;
        retrain = False;
        if("--retrain" in sys.argv):
            retrain = True;
        if(os.path.exists(w_dmp_fname) and not retrain):
            try:
                print("loading weights...", end = '');
                with open(w_dmp_fname, 'rb') as f:
                    w = pickle.load(f);
                print("Done")
            except Exception as e:
                print("read from file failed: %s" % e);
                w = np.zeros(7840).reshape(10, 784);
                print("weight reinitialized");
        else:
            w = np.zeros(7840).reshape(10, 784);
            print("weight reinitialized");
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def backup_weights(w, ):
    fn_name = "backup_weights";
    try:
        w_dmp_fname = "../dmp/w.pickle";
        with open(w_dmp_fname, 'wb') as f:
            pickle.dump(w, f);
        return True;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return False;

def update_weights(w, x, score, learning_rate, backup = False, ):
    fn_name = "update_weights";
    try:
        X = x[0].reshape(1, -1);
        print(X.shape);
        gt = x[1];
        prob = score;
        prob -= np.max(prob);
        prob = np.exp(prob) / np.sum(np.exp(prob));
        dL_dw = np.dot(prob, X);
        dL_dw[gt] = dL_dw[gt] - X;
        w -= learning_rate * dL_dw;
        if(backup):
            backup_weights(w);
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def train(training_imgs, w, learning_rate, ):
    fn_name = "train";
    try:
        YES = 0;
        NO = 0;
        size = len(training_imgs);
        for i in range(size):
            score = np.dot(w, training_imgs[i][0]);
            
            idx = np.argmax(score);
            if(idx == training_imgs[i][1]):
                YES += 1;
            else:
                NO += 1;
            ratio = 100 * YES/(YES+NO);
            if(i % 1000 == 999):
                print("\rtrained \033[1;37m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%" % (YES + NO, YES, NO, ratio), end = '     ');
                # print("\rtrained \033[1;34m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%, (%g, %g)" % (YES + NO, YES, NO, ratio, np.amax(w), np.amin(w)), end = '     ');
            update_weights(w, training_imgs[i], score, learning_rate, i % 1000 == 999);
        return YES / size;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def refresh_log():
    fn_name = "refresh_log";
    try:
        fname = ".training_precision.log";
        with open(fname, 'w') as f:
            f.write("");
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def log_precision(precision, learning_rate, ):  # precision ranges [0, 1]
    fn_name = "log_precision";
    try:
        fname = ".training_precision.log";
        with open(fname, 'a') as f:
            f.write("%f %f\n" % (precision, learning_rate));
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        w = init_weights();
        epoch = 100;
        if("--epoch" in sys.argv):
            try:
                epoch = int(sys.argv[sys.argv.index("--epoch")+1]);
            except Exception as e:
                pass;
        print("\033[1;37mstarting training\033[0m: epoch = %d" % (epoch));
        refresh_log();
        for i in range(epoch):
            # learning_rate = 1; # constant
            # learning_rate = (epoch - i) / (epoch); # linear
            # learning_rate = 1 / (i + 1); # hyperbola
            learning_rate = 1 / (1 + np.exp(i + 1 - epoch / 2)); # sigmoid # best performance
            # learning_rate = (np.arctan(-(i+1 - epoch/2)) + np.pi/2) / np.pi; # arctan
            print("\ntraining epoch %d/%d with learning_rate=%f" % (i+1, epoch, learning_rate));
            training_imgs = load_training_set();
            precision = train(training_imgs, w, learning_rate);
            backup_weights(w);
            log_precision(precision, learning_rate);
            print();
    except Exception as e:
        print("%s(): %s" % (fn_name, e));


if(__name__ == "__main__"):
    main();
