__author__ = "Blurgy";
"""
layers.py
    conv_layer 
    bottleneck 
    pooling_layer 
    bn_layer_fc 
    bn_layer_conv 
    fc_layer 
    ReLU 
    dropout_layer 
"""

import numpy as np 
import nn_utils as nn 

class conv_layer:
    def __init__(self, k_filter, f_size, f_depth, padding, stride, ):
        # k: number of filters
        # f: filters' spatial extent
        # stride = 1
        # padding = 1
        self.k_filter = k_filter;
        self.f_size = f_size;
        self.f_depth = f_depth;
        self.stride = stride;
        self.padding = padding;
        self.init_filters();
        self.init_bias();
        self.t = 0;
    def init_filters(self, ):
        self.filters = np.random.randn(self.k_filter, self.f_depth*self.f_size*self.f_size) / np.sqrt(self.f_size*self.f_size*self.f_depth);
        self.f_m, self.f_v = 0, 0;
    def init_bias(self, ):
        self.b = 0.01 * np.random.randn(self.k_filter, 1);
        self.b_m, self.b_v = 0, 0;
    def forward(self, x, ):
        self.x = x;
        N, C, H, W = self.x.shape
        self.x_col = nn.im2col(self.x, filter_h = self.f_size, filter_w = self.f_size, 
                          padding = self.padding, stride = self.stride);
        # self.x_col.shape: (fh*fw*fd, positions)
        # self.filters.shape: (k_filter, fh*fw*d)
        self.z = self.filters @ self.x_col + self.b;
        # self.z.shape: (k_filter, positions)
        out_h = int((H + 2 * self.padding - self.f_size) / self.stride + 1);
        out_w = int((W + 2 * self.padding - self.f_size) / self.stride + 1);
        self.z = self.z.reshape(self.k_filter, out_h, out_w, N); # ...
        self.z = self.z.transpose(3, 0, 1, 2);
        # self.z.shape: (N, k_filter, out_h, out_w)
        return self.z;
    def backward(self, dz, ):
        # dz.shape: (N, k_filter, out_h, out_w)
        self.db = np.sum(dz, axis=(0, 2, 3));
        # self.db.shape: (self.k_filter, 1);
        dz_reshaped = dz.transpose(1, 2, 3, 0).reshape(self.k_filter, -1);
        self.df = dz_reshaped @ self.x_col.T;
        self.df = self.df.reshape(self.filters.shape);
        self.dx_col = self.filters.T @ dz_reshaped;
        self.dx = nn.col2im(self.dx_col, self.x.shape, 
                         filter_h = self.f_size, filter_w = self.f_size, 
                         padding = self.padding, stride = self.stride);
        N = self.x.shape[0];
        self.db = self.db.reshape(self.k_filter, 1) / N;
        self.df = self.df / N;
        return self.dx;
    def update(self, learning_rate, ):
        assert self.b.size == self.db.size 
        self.t += 1;
        beta_1, beta_2 = 0.9, 0.999;
        eps = 1e-2;
        self.f_m = beta_1*self.f_m + (1-beta_1)*self.df;
        self.f_v = beta_2*self.f_v + (1-beta_2)*np.power(self.df, 2);
        self.b_m = beta_1*self.b_m + (1-beta_1)*self.db;
        self.b_v = beta_2*self.b_v + (1-beta_2)*np.power(self.db, 2);
        learning_rate = learning_rate * np.sqrt(1-np.power(beta_2, self.t)) / (1-np.power(beta_1, self.t));
        # print(" --", np.max(np.abs(-learning_rate * self.f_m / (np.sqrt(self.f_v) + eps))), 
        #              np.max(np.abs(self.x_col.T)),
        #              self.x_col.T.shape);
        self.filters += -learning_rate * self.f_m / (np.sqrt(self.f_v) + eps);
        self.b += -learning_rate * self.b_m / (np.sqrt(self.b_v) + eps);

class bottleneck:
    def __init__(self, k_filter, f_size, f_depth, padding, stride, ):
        C_reduced = np.ceil((f_depth-1) / 2);
        k_filter_reduced = np.ceil((k_filter-1) / 2);
        C_reduced, k_filter_reduced = int(C_reduced), int(k_filter_reduced);
        C_reduced = max(1, C_reduced);
        k_filter_reduced = max(1, k_filter_reduced);
        self.bottleneck_1 = conv_layer(k_filter = C_reduced, 
                                       f_size = 1, f_depth = f_depth, 
                                       padding = 0, stride = stride);
        self.relu_1 = ReLU();
        self.conv = conv_layer(k_filter = k_filter_reduced, 
                               f_size = f_size, f_depth = C_reduced, 
                               padding = padding, stride = stride);
        self.relu_2 = ReLU();
        self.bottleneck_2 = conv_layer(k_filter = k_filter, 
                                       f_size = 1, f_depth = k_filter_reduced, 
                                       padding = 0, stride = 1);
    def forward(self, x, ):
        x = self.bottleneck_1.forward(x);
        x = self.relu_1.forward(x);
        x = self.conv.forward(x);
        x = self.relu_2.forward(x);
        x = self.bottleneck_2.forward(x);
        return x;
    def backward(self, dz, ):
        dz = self.bottleneck_2.backward(dz);
        dz = self.relu_2.backward(dz);
        dz = self.conv.backward(dz);
        dz = self.relu_1.backward(dz);
        dz = self.bottleneck_1.backward(dz);
        return dz;
    def update(self, learning_rate, ):
        self.bottleneck_1.update(learning_rate);
        self.bottleneck_2.update(learning_rate);
        self.conv.update(learning_rate);

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
        self.z = self.z.transpose(2, 3, 0, 1);
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

class bn_layer_fc:
    def __init__(self, n_neuron, ):
        self.n_neuron = n_neuron;
        self.running_mean = 0;
        self.running_var = 0;
        self.momentum = 0.9;
        self.init_gamma();
        self.init_beta();
        self.eps = 1e-8;
        self.t = 0;
    def init_gamma(self, ):
        self.gamma = np.random.randn(self.n_neuron).reshape(-1,1) / np.sqrt(self.n_neuron/2);
        self.gamma_m, self.gamma_v = 0, 0;
    def init_beta(self, ):
        self.beta = np.random.randn(self.n_neuron).reshape(-1,1) / np.sqrt(self.n_neuron);
        self.beta_m, self.beta_v = 0, 0;
    def forward(self, x, is_test_time, ):
        if(not is_test_time):
            self.mean = np.mean(x, axis=(0,2)).reshape(-1,1);
            self.var = np.var(x, axis=(0,2)).reshape(-1,1);
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean;
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var;
        else:
            self.mean = self.running_mean;
            self.var = self.running_var;
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps);
        self.z = self.gamma * self.x_hat + self.beta;
        return self.z;
    def backward(self, dz, ):
        self.dx = self.gamma / np.sqrt(self.var + self.eps) * dz;
        self.dgamma = np.sum(dz * self.x_hat, axis=(0,2)).reshape(-1,1);
        self.dbeta = np.sum(dz, axis=(0,2)).reshape(-1,1);
        N = self.x_hat.shape[0];
        self.dgamma /= N;
        self.dbeta /= N;
        return self.dx;
    def update(self, learning_rate, ):
        self.t += 1;
        beta_1, beta_2 = 0.9, 0.999;
        eps = 1e-2;
        self.gamma_m = beta_1*self.gamma_m + (1-beta_1)*self.dgamma;
        self.gamma_v = beta_2*self.gamma_v + (1-beta_2)*np.power(self.dgamma, 2);
        self.beta_m = beta_1*self.beta_m + (1-beta_1)*self.dbeta;
        self.beta_v = beta_2*self.beta_v + (1-beta_2)*np.power(self.dbeta, 2);
        learning_rate = learning_rate * np.sqrt(1-np.power(beta_2, self.t)) / (1 - np.power(beta_1, self.t));
        self.gamma += -learning_rate * self.gamma_m / (np.sqrt(self.gamma_v) + eps);
        self.beta += -learning_rate * self.beta_m / (np.sqrt(self.beta_v) + eps);

class bn_layer_conv:
    def __init__(self, C, ):
        self.C = C;
        self.running_mean = 0;
        self.running_var = 0;
        self.momentum = 0.9;
        self.init_gamma();
        self.init_beta();
        self.eps = 1e-8;
        self.t = 0;
    def init_gamma(self, ):
        self.gamma = np.random.randn(self.C).reshape(-1,1,1) / np.sqrt(self.C/2);
        self.gamma_m, self.gamma_v = 0, 0;
    def init_beta(self, ):
        self.beta = np.random.randn(self.C).reshape(-1,1,1) / np.sqrt(self.C);
        self.beta_m, self.beta_v = 0, 0;
    def forward(self, x, is_test_time, ):
        if(not is_test_time):
            self.mean = np.mean(x, axis=(0,2,3)).reshape(-1,1,1);
            self.var = np.var(x, axis=(0,2,3)).reshape(-1,1,1);
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean;
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var;
        else:
            self.mean = self.running_mean;
            self.var = self.running_var;
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps);
        self.z = self.gamma * self.x_hat + self.beta;
        return self.z;
    def backward(self, dz, ):
        self.dx = self.gamma / np.sqrt(self.var + self.eps) * dz;
        self.dgamma = np.sum(dz * self.x_hat, axis=(0,2,3)).reshape(-1,1,1);
        self.dbeta = np.sum(dz, axis=(0,2,3)).reshape(-1,1,1);
        N = self.x_hat.shape[0];
        self.dgamma /= N;
        self.dbeta /= N;
        return self.dx;
    def update(self, learning_rate, ):
        self.t += 1;
        beta_1, beta_2 = 0.9, 0.999;
        eps = 1e-2;
        self.gamma_m = beta_1*self.gamma_m + (1-beta_1)*self.dgamma;
        self.gamma_v = beta_2*self.gamma_v + (1-beta_2)*np.power(self.dgamma, 2);
        self.beta_m = beta_1*self.beta_m + (1-beta_1)*self.dbeta;
        self.beta_v = beta_2*self.beta_v + (1-beta_2)*np.power(self.dbeta, 2);
        learning_rate = learning_rate * np.sqrt(1-np.power(beta_2, self.t)) / (1 - np.power(beta_1, self.t));
        self.gamma += -learning_rate * self.gamma_m / (np.sqrt(self.gamma_v) + eps);
        self.beta += -learning_rate * self.beta_m / (np.sqrt(self.beta_v) + eps);

class fc_layer:
    def __init__(self, input_size, output_size, ):
        self.input_size = input_size;
        self.output_size = output_size;
        self.init_weights();
        self.init_bias();
        self.t = 0;
    def init_weights(self, ):
        self.w = np.random.randn(self.output_size, self.input_size) / np.sqrt(self.input_size/2);
        self.w_m, self.w_v = 0, 0;
    def init_bias(self):
        self.b = np.random.randn(self.output_size, 1) / np.sqrt(self.input_size);
        self.b_m, self.b_v = 0, 0;
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
        self.dw = self.dw / N;
        self.db = self.db / N;
        return self.dx;
    def update(self, learning_rate, ):
        self.t += 1;
        beta_1, beta_2 = 0.9, 0.999;
        eps = 1e-2;
        self.w_m = beta_1*self.w_m + (1-beta_1)*self.dw;
        self.w_v = beta_2*self.w_v + (1-beta_2)*np.power(self.dw, 2);
        self.b_m = beta_1*self.b_m + (1-beta_1)*self.db;
        self.b_v = beta_2*self.b_v + (1-beta_2)*np.power(self.db, 2);
        learning_rate = learning_rate * np.sqrt(1-np.power(beta_2, self.t)) / (1 - np.power(beta_1, self.t));
        self.w += -learning_rate * self.w_m / (np.sqrt(self.w_v) + eps);
        self.b += -learning_rate * self.b_m / (np.sqrt(self.b_v) + eps);

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
