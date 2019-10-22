__author__ = "Blurgy";

import os 
import pickle 
import numpy as np 
from scipy import ndimage 

def shift(x):
    flag = np.random.randint(0,3);
    dist = np.random.randint(1,3);
    dist *= 1 if np.random.randn() > 0 else -1;
    if(flag == 0):
        return np.roll(x, dist, axis=0);
    elif(flag == 1):
        return np.roll(x, dist, axis=1);
    else:
        return np.roll(x, dist, axis=(0,1));

def rotate(x, deg):
    ret = x.reshape(28,28);
    ret = ndimage.rotate(ret.astype('uint8'), deg, reshape=False);
    return ret.astype(x.dtype);

def zoom(x, ratio):
    ret = x.reshape(28,28);
    ret = ndimage.zoom(ret.astype('uint8'), ratio);
    if(ratio > 1):
        ret = ret[1:-1, 1:-1];
    else:
        ret = np.pad(ret, ((1,1), (1,1)));
    return ret.astype(x.dtype);

def augment_training_set(train):
    ret = train.copy();
    for elem in train:
        label, img = elem;
        img = img.reshape(28, 28);
        # (h/v) shift (1/2) grids (random) 
        ret.append([label, shift(img).reshape(1,28,28)]);
        # rotate 10~15 degrees 
        deg = np.random.randint(10,16);
        deg *= 1 if np.random.randn() > 0 else -1;
        ret.append([label, rotate(img, deg).reshape(1,28,28)]);
        # zoom +- 7% 
        ratio = 0.07;
        ratio *= 1 if np.random.randn() > 0 else -1;
        ret.append([label, zoom(img, 1+ratio).reshape(1,28,28)]);
    return ret;

def normalize_training_set(train):
    for elem in train:
        label, img = elem;
        img -= int(np.mean(img));

def normalize_testing_set(test):
    for img in test:
        img -= int(np.mean(img));

def load_training_set():
    fpath = os.path.join(os.getcwd(), "dat", "train.csv");
    ret = [];
    with open(fpath) as f:
        f.readline();
        for line in f.readlines():
            data = line.strip().split(',');
            label = int(data[0].strip());
            imginfo = [];
            for i in range(1, len(data)):
                imginfo.append(int(data[i].strip()));
            imginfo = np.array(imginfo).reshape(1,28,28).astype('int16');
            ret.append([label, imginfo]);
            if(len(ret) % 1000 == 0):
                print("\rloading training set: %d" % (len(ret)), end = "");
    print()
    return ret;

def load_testing_set():
    fpath = os.path.join(os.getcwd(), "dat", "test.csv");
    ret = [];
    with open(fpath) as f:
        f.readline();
        for line in f.readlines():
            data = line.strip().split(',');
            imginfo = [];
            for i in range(0, len(data)):
                imginfo.append(int(data[i].strip()));
            imginfo = np.array(imginfo).reshape(1,28,28).astype('int16');
            ret.append(imginfo);
            if(len(ret) % 1000 == 0):
                print("\rloading testing set: %d" % (len(ret)), end = "");
    print("\ncomplete\n");
    return ret;

def preprocess_training_set(aug = True):
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
    if(aug):
        # augment
        print("augmenting.. ", end="");
        training_set = augment_training_set(training_set);
    print("normalizing.. ", end="");
    normalize_training_set(training_set);
    print("complete, size %d" % (len(training_set)));
    np.random.shuffle(training_set);
    return training_set;

def preprocess_testing_set():
    dmp_path = os.path.join("dmp", "test.pickle");
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    testing_set = None;
    if(not os.path.exists(dmp_path)):
        testing_set = load_testing_set();
        normalize_testing_set(testing_set);
        with open(dmp_path, 'wb') as f:
            pickle.dump(testing_set, f);
    else:
        with open(dmp_path, 'rb') as f:
            testing_set = pickle.load(f);
    return testing_set;

def sample_batches_train(training_set, batch_size):
    X, Y, x, y = [], [], [], [];
    cnt = 0;
    for elem in training_set:
        label, imginfo = elem;
        x.append(imginfo);
        y.append(label);
        cnt += 1;
        if(cnt == batch_size):
            X.append(np.array(x));
            Y.append(np.array(y));
            cnt = 0;
            x = [];
            y = [];
    if(cnt > 0):
        X.append(np.array(x));
        Y.append(np.array(y));
    return X, Y;

def sample_batches_test(testing_set, batch_size):
    X, x = [], [];
    cnt = 0;
    for elem in testing_set:
        x.append(elem);
        cnt += 1;
        if(cnt == batch_size):
            X.append(np.array(x));
            cnt = 0;
            x = [];
    if(cnt > 0):
        X.append(np.array(x));
    return X;

def save_model(model):
    model_dmp_path = os.path.join("dmp", model['name']+".pickle");
    if(not os.path.exists("dmp")):
        os.makedirs("dmp");
    with open(model_dmp_path, 'wb') as f:
        pickle.dump(model, f);

def load_model(ipath):
    model_dmp_path = ipath;
    if(not os.path.exists(model_dmp_path)):
        raise ValueError("file [%s] does not exist" % model_dmp_path);
    model = None;
    with open(model_dmp_path, 'rb') as f:
        model = pickle.load(f);
    return model;
