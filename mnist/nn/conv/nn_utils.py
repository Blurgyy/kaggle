__author__ = "Blurgy";
"""
nn_utils.py
    get_im2col_indices 
    im2col 
    col2im 
    decay_schedule 
    sample_batches 
    init_model 
    forward 
    backward 
    grad 
    update 
"""

import numpy as np 
from layers import * 

def get_im2col_indices(x_shape, filter_h, filter_w, padding, stride):
    N, C, H, W = x_shape
    out_h = int((H + 2 * padding - filter_h) / stride + 1)
    out_w = int((W + 2 * padding - filter_h) / stride + 1)

    i0 = np.repeat(np.arange(filter_h), filter_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)

    j0 = np.tile(np.arange(filter_w), filter_h * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    k = np.repeat(np.arange(C), filter_h * filter_w).reshape(-1, 1);
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (k, i, j)


def im2col(x, filter_h, filter_w, padding, stride):
    N, C, H, W = x.shape
    k, i, j = get_im2col_indices(x.shape, filter_h, filter_w, padding, stride)
    p = padding
    x_pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)))
    cols = x_pad[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(filter_h*filter_w*C, -1)
    return cols

def col2im(cols, x_shape, filter_h, filter_w, padding, stride):
    N, C, H, W = x_shape
    p = padding
    H_pad, W_pad = H + 2 * p, W + 2 * p
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype = cols.dtype)
    k, i, j = get_im2col_indices(x_shape, filter_h, filter_w, p, stride)
    cols_reshaped = cols.reshape(C * filter_h * filter_w, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_pad, (slice(None), k, i, j), cols_reshaped)
    if(p == 0):
        return x_pad
    return x_pad[:, :, p:-p, p:-p]

def decay_schedule(length, name):
    i = np.arange(length);
    if(name == "sigmoid"):
        return 1 / (1 + np.exp(i + 1 - length / 2));
    elif(name == "linear"):
        return (length - i) / length;
    elif(name == "hyperbola"):
        return 1 / (i + 1);
    elif(name == "constant"):
        return np.ones(length);
    elif(name == "exponential"):
        # return 0.995 ** (i); # without batch-norm
        return 0.97 ** (i); # with batch-norm

def init_model(C1, C2, C3, C4):
    """
    [conv -> batch-norm -> relu]*2 -> pool -> 
    -> conv -> batch-norm -> relu -> pool -> 
    -> fc6 -> batch-norm -> relu -> fc7 -> 
    -> output
    """
    model = {};
    model['conv1'] = bottleneck(k_filter = C1,
                                f_size = 3, f_depth = 1, 
                                padding = 1, stride = 1);
    model['bn1'] = bn_layer_conv(C1);
    model['relu1'] = ReLU();
    model['conv2'] = bottleneck(k_filter = C2,
                                f_size = 3, f_depth = C1, 
                                padding = 1, stride = 1);
    model['bn2'] = bn_layer_conv(C2);
    model['relu2'] = ReLU();
    model['pooling1'] = pooling_layer(size = 2, padding = 0, stride = 2);
    model['conv3'] = bottleneck(k_filter = C3,
                                f_size = 3, f_depth = C2, 
                                padding = 1, stride = 1);
    model['bn3'] = bn_layer_conv(C3);
    model['relu3'] = ReLU();
    model['conv4'] = bottleneck(k_filter = C4,
                                f_size = 3, f_depth = C3, 
                                padding = 1, stride = 1);
    model['bn4'] = bn_layer_conv(C4);
    model['relu4'] = ReLU();
    model['pooling2'] = pooling_layer(size = 2, padding = 0, stride = 2);
    model['fc6'] = fc_layer(input_size = C4*7*7, output_size = 1024);
    model['bn5'] = bn_layer_fc(1024);
    model['relu5'] = ReLU();
    model['fc7'] = fc_layer(input_size = 1024, output_size = 10);
    model['output'] = None;
    return model;

def forward(model, x, is_test_time):
    x = model['conv1'].forward(x);
    x = model['bn1'].forward(x, is_test_time);
    x = model['relu1'].forward(x);
    x = model['conv2'].forward(x);
    x = model['bn2'].forward(x, is_test_time);
    x = model['relu2'].forward(x);
    x = model['pooling1'].forward(x);
    x = model['conv3'].forward(x);
    x = model['bn3'].forward(x, is_test_time);
    x = model['relu3'].forward(x);
    x = model['conv4'].forward(x);
    x = model['bn4'].forward(x, is_test_time);
    x = model['relu4'].forward(x);
    x = model['pooling2'].forward(x);
    N, C, H, W = x.shape;
    x = x.reshape(N, -1, 1);
    x = model['fc6'].forward(x);
    x = model['bn5'].forward(x, is_test_time);
    x = model['relu5'].forward(x);
    model['output'] = model['fc7'].forward(x);
    return model['output'];

def backward(model, dz):
    dz = model['fc7'].backward(dz);
    dz = model['relu5'].backward(dz);
    dz = model['bn5'].backward(dz);
    dz = model['fc6'].backward(dz);
    # N, C, H, W = model['pooling2'].z.shape;
    # dz = dz.reshape(N, C, H, W);
    dz = dz.reshape(model['pooling2'].z.shape);

    dz = model['pooling2'].backward(dz);
    dz = model['relu4'].backward(dz);
    dz = model['bn4'].backward(dz);
    dz = model['conv4'].backward(dz);
    dz = model['relu3'].backward(dz);
    dz = model['bn3'].backward(dz);
    dz = model['conv3'].backward(dz);
    dz = model['pooling1'].backward(dz);
    dz = model['relu2'].backward(dz);
    dz = model['bn2'].backward(dz);
    dz = model['conv2'].backward(dz);
    dz = model['relu1'].backward(dz);
    dz = model['bn1'].backward(dz);
    model['conv1'].backward(dz);

def grad(model, y):
    prob = model['output'].copy();
    prob -= np.max(prob, axis=1).reshape(-1,1,1);
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1).reshape(-1,1,1);
    dz = prob.copy();
    a0 = np.arange(len(y)).reshape(-1,1);
    y = y.reshape(-1,1);
    a2 = np.repeat(0, len(y)).reshape(-1,1);
    np.add.at(dz, (a0,y,a2), -1);
    try:
        loss = np.sum(-np.log(prob[a0,y,a2]))
    except RuntimeWarning:
        print("\n-- RuntimeWarning in log():");
        print(prob[a0,y,a2]);
        exit();
    return dz, loss;

def update(model, learning_rate):
    model['conv1'].update(learning_rate);
    model['conv2'].update(learning_rate);
    model['conv3'].update(learning_rate);
    model['conv4'].update(learning_rate);
    model['bn1'].update(learning_rate);
    model['bn2'].update(learning_rate);
    model['bn3'].update(learning_rate);
    model['bn4'].update(learning_rate);
    model['bn5'].update(learning_rate);
    model['fc6'].update(learning_rate);
    model['fc7'].update(learning_rate);
