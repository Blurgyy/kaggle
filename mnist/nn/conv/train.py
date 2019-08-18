#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import nn_utils as nn 
import plot 
import click 

import warnings
warnings.filterwarnings("error")

@click.command()
@click.option("--epoch", type = int, default = 10, 
              help = "Specifies number of epoches, 10 by default")
@click.option("--rate", type = float, default = 1, 
              help = "Specifies value of initial learning rate, 1 by default")
@click.option("--reg", type = float, default = 1e-3,
              help = "Specifies value of [regularization strenth]/[learning rate], 1e-3 by default")
@click.option("--decay", type = click.Choice(["exponential", "constant", "linear", "sigmoid", "hyperbola"]), 
              default = "exponential", 
              help = "Specifies decay schedule of learning rate, exponential by default")
@click.option("--continue-at", type = click.Path(exists=True), default = None, 
              help = "Continues training at specified file, initializes a new model if not specified")
@click.option("--batch-size", type = int, default = 32, 
              help = "Specifies batch size, 32 by default")
def main(epoch, rate, reg, decay, continue_at, batch_size):
    base_learning_rate = rate;
    decay_schedule = nn.decay_schedule(epoch, decay);
    learning_rate = base_learning_rate * decay_schedule;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model(reg);

    for ep in range(epoch):
        lr = learning_rate[ep];
        print("epoch %d/%d, batch size %d, learning rate %g" % (ep+1, epoch, batch_size, lr))
        train = data.preprocess_training_set();
        X, Y = data.sample_batches_train(train, batch_size);
        del train;
        print("training set loaded and shuffled");

        yes, cnt, epoch_loss = 0, 0, 0;
        for i in range(len(X)):
            x, y = X[i], Y[i];
            nn.forward(model, x, is_test_time = False);
            dz, loss = nn.grad(model, y);
            epoch_loss += loss;
            nn.backward(model, dz);

            prediction = np.argmax(model['output'], axis=1);
            score = prediction.reshape(-1,1) == y.reshape(-1,1);
            yes += np.sum(score);
            cnt += len(y);
            print("[%d/%d]: %.2f%%, batch loss: %.2f    " % (yes, cnt, yes / cnt * 100, loss), end = '\r');
            nn.update(model, lr);
        print();
        print("epoch %d/%d, overall loss %.2f" % (ep+1, epoch, epoch_loss));
        data.save_model(model);
        print("model saved\n");

if(__name__ == "__main__"):
    main();