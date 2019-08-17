"""
layers.py
    conv_layer 
    pooling_layer 
    bn_layer 
    fc_layer 
    ReLU 
    dropout_layer 
"""

import numpy as np 
import nn_utils as nn 

class conv_layer:
    def __init__(self, k_filters, f_size, f_depth, padding, stride):
        # k: number of filters
        # f: filters' spatial extent
        # stride = 1
        # padding = 1
        self.k_filters = k_filters;
        self.f_size = f_size;
        self.f_depth = f_depth;
        self.stride = stride;
        self.padding = padding;
        self.init_filters();
        self.init_bias();
    def init_filters(self, ):
        self.filters = np.random.randn(self.k_filters, self.f_depth*self.f_size*self.f_size) / np.sqrt(self.f_size*self.f_size*self.f_depth);
        self.f_v = 0;
    def init_bias(self, ):
        self.b = 0.01 * np.random.randn(self.k_filters, 1);
        self.b_v = 0;
    def forward(self, x, ):
        self.x = x;
        N, C, H, W = self.x.shape
        self.x_col = nn.im2col(self.x, filter_h = self.f_size, filter_w = self.f_size, 
                          padding = self.padding, stride = self.stride);
        # self.x_col.shape: (fh*fw*fd, positions)
        # self.filters.shape: (k_filters, fh*fw*d)
        self.z = self.filters @ self.x_col + self.b;
        # self.z.shape: (k_filters, positions)
        out_h = int((H + 2 * self.padding - self.f_size) / self.stride + 1);
        out_w = int((W + 2 * self.padding - self.f_size) / self.stride + 1);
        self.z = self.z.reshape(self.k_filters, out_h, out_w, N); # ...
        self.z = self.z.transpose(3, 0, 1, 2);
        # self.z.shape: (N, k_filters, out_h, out_w)
        return self.z;
    def backward(self, dz, ):
        # dz.shape: (N, k_filters, out_h, out_w)
        self.db = np.sum(dz, axis=(0, 2, 3));
        # self.db.shape: (self.k_filters, 1);
        dz_reshaped = dz.transpose(1, 2, 3, 0).reshape(self.k_filters, -1);
        self.df = dz_reshaped @ self.x_col.T;
        self.df = self.df.reshape(self.filters.shape);
        self.dx_col = self.filters.T @ dz_reshaped;
        self.dx = nn.col2im(self.dx_col, self.x.shape, 
                         filter_h = self.f_size, filter_w = self.f_size, 
                         padding = self.padding, stride = self.stride);
        return self.dx;
    def update(self, learning_rate, ):
        assert self.b.size == self.db.size
        N = self.x.shape[0];
        mu = 0.5;
        self.db = self.db.reshape(self.k_filters, 1) / N;
        self.df = self.df / N;
        self.b_v = mu * self.b_v - learning_rate * self.db;
        self.f_v = mu * self.f_v - learning_rate * self.df;
        self.b += self.b_v;
        self.filters += self.f_v;

class pooling_layer:
    def __init__(self, size, padding, stride, ):
        self.size = size;
        self.padding = padding;
        self.stride = stride;
    def forward(self, x, ):
        self.x = x;
        N, C, H, W = self.x.shape;
        out_h = int((H + 2 * self.padding - self.size) / self.stride + 1);
        out_w = int((W + 2 * self.padding - self.size) / self.stride + 1);
        x_reshaped = x.reshape(N*C, 1, H, W);
        self.x_col = nn.im2col(x_reshaped, filter_h = self.size, filter_w = self.size, 
                                   padding = self.padding, stride = self.stride);
        # self.x_col.shape: (fh*fw*d, positions)
        self.max_idx = np.argmax(self.x_col, axis=0);
        self.z = self.x_col[self.max_idx, range(self.max_idx.size)];
        self.z = self.z.reshape(out_h, out_w, N, C);
        self.z = self.z.transpose(2, 3, 0, 1)
        return self.z;
    def backward(self, dz, ):
        N, C, H, W = self.x.shape
        dx_col = np.zeros_like(self.x_col);
        dz = dz.transpose(2, 3, 0, 1); # ...
        dz_flat = dz.ravel();
        dx_col[self.max_idx, range(self.max_idx.size)] = dz_flat;
        self.dx = nn.col2im(dx_col, (N*C, 1, H, W), 
                         filter_h = self.size, filter_w = self.size, 
                         padding = self.padding, stride = self.stride);
        self.dx = self.dx.reshape(self.x.shape);
        return self.dx;

class bn_layer:
    def __init__(self, ):
        pass;
    def forward(self, x, ):
        pass;
    def backward(self, dz, ):
        pass;
    def update(self, learning_rate, ):
        pass;

class fc_layer:
    def __init__(self, input_size, output_size, reg, ):
        self.input_size = input_size;
        self.output_size = output_size;
        self.reg = reg;
        self.init_weights();
        self.init_bias();
    def init_weights(self, ):
        self.w = np.random.randn(self.output_size, self.input_size) / np.sqrt(self.input_size/2);
        self.w_v = 0;
    def init_bias(self):
        self.b = np.random.randn(self.output_size, 1) / np.sqrt(self.input_size);
        self.b_v = 0;
    def forward(self, x, ):
        self.x = x;
        N = self.x.shape[0];
        x_reshaped = self.x.transpose(2, 1, 0).reshape(self.input_size, N);
        try:
            self.z = self.w @ x_reshaped + self.b;
        except RuntimeWarning:
            print("\n-- RuntimeWarning in dot product\n:");
            print(np.max(self.w));
            print(np.max(x_reshaped));
            exit();
        self.z = self.z.T.reshape(N, self.output_size, 1);
        return self.z;
    def backward(self, dz, ):
        # print("\nfc_layer::backward()")
        N = self.x.shape[0];
        self.db = np.sum(dz, axis=(0, 2));
        self.db = self.db.reshape(self.output_size, -1)
        dz_reshaped = dz.transpose(2, 1, 0).reshape(self.output_size, N);
        x_reshaped = self.x.transpose(2, 0, 1).reshape(N, self.input_size);
        self.dw = dz_reshaped @ x_reshaped;
        self.dx = self.w.T @ dz_reshaped;
        self.dx = self.dx.T.reshape(N, self.input_size, 1);
        return self.dx;
    def update(self, learning_rate, ):
        N = self.x.shape[0];
        mu = 0.5;
        self.dw = self.dw / N;
        self.db = self.db / N;
        self.w += -self.reg * learning_rate * self.w;
        self.w_v = mu * self.w_v - learning_rate * self.dw;
        self.b_v = mu * self.b_v - learning_rate * self.db;
        self.w += self.w_v;
        self.b += self.b_v;

class ReLU:
    def __init__(self, ):
        pass; # nothing to do
    def forward(self, x):
        self.mask = (x > 0.01);
        self.z = x * self.mask;
        return self.z;
    def backward(self, dz, ):
        self.dx = dz * self.mask;
        return self.dx;

class dropout_layer:
    def __init__(self, p = 0.5, ):
        self.p = p; # dropout ratio
    def forward(self, x, is_test_time, ):
        if(is_test_time):
            self.mask = self.p;
        else:
            self.mask = (np.random.rand(*x.shape) < self.p);
        self.z = x * self.mask;
        return self.z;
    def backward(self, dz, ):
        self.dx = dz * self.mask;
        return self.dx;
