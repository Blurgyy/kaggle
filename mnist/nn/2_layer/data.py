#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import pickle 
import random 
import numpy as np 

pwd = os.getcwd();

def load_training_set():
    fpath = os.path.join(pwd, "dat/train.csv");
    ret = [];
    with open(fpath) as f:
        f.readline();
        for line in f.readlines():
            data = line.strip().split(',');
            label = int(data[0].strip());
            imginfo = [];
            for i in range(1, len(data)):
                imginfo.append(int(data[i].strip()));
            imginfo = np.array(imginfo).reshape(-1, 1);
            ret.append([label, imginfo]);
            if(len(ret) % 1000 == 0):
                print(len(ret));
    return ret;

def load_testing_set():
    fpath = os.path.join(pwd, "dat/test.csv");
    ret = [];
    with open(fpath) as f:
        f.readline();
        data = line.strip.split(',');
        imginfo = [];
        for i in range(0, len(data)):
            imginfo.append(int(data[i].strip()));
        imginfo = np.array(imginfo).reshape(-1, 1);
        ret.append(imginfo);
    return ret;

def preprocess_train():
    dmp_path = "dmp/train.pickle";
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    training_set = None;
    if(not os.path.exists(dmp_path)):
        training_set = data.load_training_set();
        with open(dmp_path, 'wb') as f:
            pickle.dump(training_set, f);
    else:
        with open(dmp_path, 'rb') as f:
            training_set = pickle.load(f);
    random.shuffle(training_set);
    return training_set;

def save_model(model):
    model_dmp_path = "dmp/model.pickle";
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    with open(model_dmp_path, 'wb') as f:
        pickle.dump(model, f);

def load_model():
    model_dmp_path = "dmp/model.pickle";
    if(not os.path.exists(model_dmp_path)):
        raise ValueError("file [%s] does not exist!" % model_dmp_path);
    model = None;
    with open(model_dmp_path, 'rb') as f:
        modle = pickle.load(f);
    return model;
