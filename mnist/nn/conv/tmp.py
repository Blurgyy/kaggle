#!/usr/bin/python3

import numpy as np 
import nn 

x = np.random.randn(1, 28, 280);
layer = nn.conv_layer(k_filters = 2,
					  f_size = 3,
					  f_depth = 1);

print(x.shape);
y = layer.forward(x);
print(y.shape)
