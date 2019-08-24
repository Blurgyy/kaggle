__author__ = "Blurgy";

import os 
import pickle 
import numpy as np 
from PIL import Image 

def shift(x):
    flag = np.random.randint(0,2);
    dist = np.random.randint(1,2);
    if(flag == 0):
        return np.roll(x, dist, axis=0);
    elif(flag == 1):
        return np.roll(x, dist, axis=1);
    else:
        return np.roll(x, dist, axis=(0,1));

def rotate(x, deg):
    ret = x.reshape(28,28);
    img = Image.fromarray(ret.astype('uint8'));
    rot = img.rotate(deg);
    return np.asarray(rot);

def augment_training_set(train):
    ret = train.copy();
    for elem in train:
        label, img = elem;
        img = img.reshape(28, 28);
        # (h/v) shift (1/2) grids (random)
        ret.append([label, shift(img).reshape(1,28,28)]);
        # rotate 5~15 degrees 
        deg = np.random.randint(5,16);
        deg *= 1 if np.random.rand() > 1 else -1;
        ret.append([label, rotate(img, deg).reshape(1,28,28)]);
        pass;
    return ret;

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
        # augmentation
        print("\naugmenting... ", end="");
        ret = augment_training_set(ret);
        for elem in ret:
            label, img = elem;
            img = img.astype('int16');
            img -= int(np.mean(img)); # zero centering
    print("complete, size %d\n" % (len(ret)));
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
            imginfo -= int(np.mean(imginfo)); # zero centering
            ret.append(imginfo);
            if(len(ret) % 1000 == 0):
                print("\rloading testing set: %d" % (len(ret)), end = "");
    print("\ncomplete\n");
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
    np.random.shuffle(training_set);
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
