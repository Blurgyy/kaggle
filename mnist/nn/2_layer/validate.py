#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import data 
import os 
import sys 
import numpy as np 

def forward(model, img):
    model['input'] = img;
    model['h1'] = model['layer_1'].forward(model['w1'], model['input']);
    model['h1'] = model['ReLU'].forward(model['h1']);
    model['score'] = model['layer_2'].forward(model['w2'], model['h1']);

def main():
	if(len(sys.argv) != 2):
		print("usage: python3 validate.py <model_dump_file>");
		exit();
	model = data.load_model(sys.argv[1]);
	testing_set = data.preprocess_testing_set();
	fpath = os.path.join("submit", "submit.csv");
	with open(fpath, 'w') as f:
		f.write("ImageId,Label");
		for i in range(len(testing_set)):
			forward(model, testing_set[i]);
			f.write("\n%d,%d" % (i+1, np.argmax(model['score'])));

if(__name__ == "__main__"):
	main();
