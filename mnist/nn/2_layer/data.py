#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import pickle 
import random 
import gates 
import numpy as np 

def load_training_set():
    fpath = os.path.join(os.getcwd(), "dat", "train.csv");
    ret = [];
    print();
    with open(fpath) as f:
        f.readline();
        for line in f.readlines():
            data = line.strip().split(',');
            label = int(data[0].strip());
            imginfo = [];
            for i in range(1, len(data)):
                imginfo.append(int(data[i].strip()));
            imginfo = np.array(imginfo).reshape(-1, 1);
            imginfo -= int(np.mean(imginfo)); # zero centering
            ret.append([label, imginfo]);
            if(len(ret) % 1000 == 0):
                print("\rloading training set: %d" % (len(ret)), end = "");
    print("\ncomplete");
    return ret;

def load_testing_set():
    fpath = os.path.join(os.getcwd(), "dat", "test.csv");
    ret = [];
    print();
    with open(fpath) as f:
        f.readline();
        for line in f.readlines():
            data = line.strip().split(',');
            imginfo = [];
            for i in range(0, len(data)):
                imginfo.append(int(data[i].strip()));
            imginfo = np.array(imginfo).reshape(-1, 1);
            imginfo -= int(np.mean(imginfo)); # zero centering
            ret.append(imginfo);
            if(len(ret) % 1000 == 0):
                print("\rloading testing set: %d" % (len(ret)), end = "");
    print("\ncomplete");
    return ret;

def preprocess_training_set():
    dmp_path = os.path.join("dmp", "train.pickle");
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    training_set = None;
    if(not os.path.exists(dmp_path)):
        training_set = load_training_set();
        with open(dmp_path, 'wb') as f:
            pickle.dump(training_set, f);
    else:
        with open(dmp_path, 'rb') as f:
            training_set = pickle.load(f);
    random.shuffle(training_set);
    return training_set;

def preprocess_testing_set():
    dmp_path = os.path.join("dmp", "test.pickle");
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    testing_set = None;
    if(not os.path.exists(dmp_path)):
        testing_set = load_testing_set();
        with open(dmp_path, 'wb') as f:
            pickle.dump(testing_set, f);
    else:
        with open(dmp_path, 'rb') as f:
            testing_set = pickle.load(f);
    # random.shuffle(testing_set);
    return testing_set;

def save_model(model):
    model_dmp_path = os.path.join("dmp", "model.pickle");
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    with open(model_dmp_path, 'wb') as f:
        pickle.dump(model, f);

def load_model(ipath = None):
    model_dmp_path = os.path.join("dmp", "model.pickle");
    if(ipath == None):
        pass;
    else:
        model_dmp_path = ipath;
    if(not os.path.exists(model_dmp_path)):
        raise ValueError("file [%s] does not exist" % model_dmp_path);
    model = None;
    with open(model_dmp_path, 'rb') as f:
        model = pickle.load(f);
    return model;
