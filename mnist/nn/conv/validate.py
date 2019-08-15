#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import data 
import nn 
import os 
import sys 
import numpy as np 
import click 

@click.command()
@click.argument("model-dump-file")
def main(model_dump_file):
    model = data.load_model(model_dump_file);
    test = np.array(data.preprocess_testing_set());
    if(not os.path.exists("submit")):
        os.makedirs("submit");
    fpath = os.path.join("submit", "submit.csv");
    with open(fpath, 'w') as f:
        f.write("ImageId,Label");
        X = data.sample_batches_test(test, 64);
        cnt = 0;
        for x in X:
            nn.forward(model, x, is_test_time = True);
            predictions = np.argmax(model['output'], axis=1).ravel();
            for i in predictions:
                cnt += 1
                f.write("\n%d,%d" % (cnt, i));

if(__name__ == "__main__"):
    main();
