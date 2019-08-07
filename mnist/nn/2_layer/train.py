#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import flow 
import click 

@click.command()
@click.option("--epoch", type = int, default = 20, 
              help = "Specifies number of epoches")
@click.option("--rate", type = float, default = 1e-5, 
              help = "Specifies value of initial learning rate")
@click.option("--descend", type = click.Choice(["linear", "sigmoid", "hyperbola"]), 
              default = "linear", 
              help = "Specifies descend pattern of learning rate")
def main(epoch, rate, descend):
    input_size = 784;
    hidden_layer_size = 200;
    output_size = 10;
    model = data.init_model(input_size, hidden_layer_size, output_size);

    # epoch = 20;
    # base_learning_rate = 1e-5;
    base_learning_rate = rate;
    descend_pattern = flow.descend_pattern(epoch, descend);
    learning_rate = base_learning_rate * descend_pattern;
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
            flow.forward(model, img);
            prob = model['score'];
            prob -= np.max(prob);
            prob = np.exp(prob) / np.sum(np.exp(prob));
            dz = prob;
            dz[label] -= 1;
            flow.backward(model, dz, lr);

            predict = np.argmax(model['score']);
            yes += (predict == label);
            cnt += 1;
            if(cnt % 1000 == 0):
                print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100), end = '\r');
        data.save_model(model);
        print("\nmodel saved\n");


if(__name__ == "__main__"):
    main();
