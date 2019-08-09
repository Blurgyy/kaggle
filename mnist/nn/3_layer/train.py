#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import nn 
import click 

@click.command()
@click.option("--epoch", type = int, default = 10, 
              help = "Specifies number of epoches, 10 by default")
@click.option("--rate", type = float, default = 1e-3, 
              help = "Specifies value of initial learning rate, 1e-3 by default")
@click.option("--reg", type = float, default = 1e-6,
              help = "Specifies regularization strenth, 1e-6 by default")
@click.option("--decay", type = click.Choice(["exponential", "constant", "linear", "sigmoid", "hyperbola"]), 
              default = "exponential", 
              help = "Specifies decay schedule of learning rate, exponential by default")
@click.option("--continue-at", type = str, default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
def main(epoch, rate, reg, decay, continue_at):
    input_size = 784;
    h1_size = 100;
    h2_size = 30;
    output_size = 10;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model(input_size, h1_size, h2_size, output_size, reg);

    # epoch = 20;
    # base_learning_rate = 1e-5;
    base_learning_rate = rate;
    decay_schedule = nn.decay_schedule(epoch, decay);
    learning_rate = base_learning_rate * decay_schedule;
    print(learning_rate);

    for ep in range(epoch):
        lr = learning_rate[ep];
        training_set = data.preprocess_training_set();
        print("training epoch %d/%d with learning rate %g" % (ep+1, epoch, lr));
        yes = 0;
        cnt = 0;
        for elem in training_set:
            label = elem[0];
            img = elem[1];
            nn.forward(model, img, is_test_time = False);
            prob = model['score'].copy();
            prob -= np.max(prob);
            prob = np.exp(prob) / np.sum(np.exp(prob));
            dz = prob.copy();
            dz[label] -= 1;
            # nn.backward(model, dz, lr);
            nn.adam_backward(model, dz, lr);

            predict = np.argmax(model['score']);
            yes += (predict == label);
            cnt += 1;
            if(cnt % 1000 == 0):
                print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100), end = '\r');
        data.save_model(model);
        print("\nmodel saved\n");


if(__name__ == "__main__"):
    main();
