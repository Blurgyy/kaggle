#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import data 
import nn_utils as nn 
import os 
import sys 
import numpy as np 
import click 

@click.command()
@click.argument("model-dump-file")
@click.option("--batch-size", type = int, default = 32, 
              help = "Specifies batch size, 32 by default")
def main(model_dump_file, batch_size):
    model = data.load_model(model_dump_file);
    test = np.array(data.preprocess_testing_set());
    if(not os.path.exists("submit")):
        os.makedirs("submit");
    fpath = os.path.join("submit", "submit.csv");
    with open(fpath, 'w') as f:
        f.write("ImageId,Label");
        X = data.sample_batches_test(test, batch_size);
        cnt = 0;
        for x in X:
            nn.forward(model, x, is_test_time = True);
            predictions = np.argmax(model['output'], axis=1).ravel();
            for i in predictions:
                cnt += 1
                if(cnt % 1000 == 0):
                    print("validating %d/%d" % (cnt,len(test)));
                f.write("\n%d,%d" % (cnt, i));

if(__name__ == "__main__"):
    main();
