#!/usr/bin/python3
# -*- coding: utf-8 -*- 
__author__ = "zgy";

import sys
import numpy as np
import pickle
import random

def load_test_set():
    fn_name = "load_test_set";
    try:
        testing_set_dmp_fname = "../dmp/testing_set.pickle";
        testing_imgs = None;
        with open(testing_set_dmp_fname, 'rb') as f:
            testing_imgs = pickle.load(f);
        # random.shuffle(testing_imgs);
        return testing_imgs;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def load_weights(fname, ):
    fn_name = "load_weights";
    try:
        w_dmp_fname = fname;
        w = None;
        with open(w_dmp_fname, 'rb') as f:
            w = pickle.load(f);
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def validate(testing_imgs, w, ):
    fn_name = "validate";
    try:
        size = len(testing_imgs);
        submit_fname = "../dat/submit.csv";
        with open(submit_fname, 'w') as f:
            pass;
        with open(submit_fname, 'a') as f:
            f.write("ImageId,Label\n");
            for i in range(size):
                x = testing_imgs[i];
                # gt = testing_imgs[i][1];
                score = np.dot(w, x);
                idx = np.argmax(score);
                f.write("%d,%d\n" % (i+1, idx));
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        testing_imgs = load_test_set();
        fname = "../dmp/w.pickle";
        if(len(sys.argv) == 2):
            fname = sys.argv[1];
        w = load_weights(fname);
        validate(testing_imgs, w);
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

if(__name__ == "__main__"):
    main();
