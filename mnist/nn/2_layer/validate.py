#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import data 
import flow 
import os 
import sys 
import numpy as np 
import click 

@click.command()
@click.argument("model-dump-file")
def main(model_dump_file):
	model = data.load_model(model_dump_file);
	testing_set = data.preprocess_testing_set();
	fpath = os.path.join("submit", "submit.csv");
	with open(fpath, 'w') as f:
		f.write("ImageId,Label");
		for i in range(len(testing_set)):
			flow.forward(model, testing_set[i]);
			f.write("\n%d,%d" % (i+1, np.argmax(model['score'])));

if(__name__ == "__main__"):
	main();
