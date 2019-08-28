#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import nn_utils as nn 
import time 
import plot 
import click 

import warnings 
warnings.filterwarnings('ignore', r'.*output shape of zoom.*')
warnings.filterwarnings('error', r'.*divide by zero.*')

@click.command()
@click.option("--epoch", type = int, default = 20, 
              help = "Specify number of epoches, 20 by default")
@click.option("--rate", type = float, default = 1e-3, 
              help = "Specify value of initial learning rate, 1e-3 by default")
@click.option("--decay", type = click.Choice(["exponential", "constant", "staircase"]), 
              default = "staircase", 
              help = "Specify decay schedule of learning rate, `staircase` by default")
@click.option("--continue-at", type = click.Path(exists=True), default = None, 
              help = "Continue training at specified file, initializes a new model if not specified")
@click.option("--batch-size", type = int, default = 32, 
              help = "Specify batch size, 32 by default")
@click.option("--channels", type = (int, int, int, int), default = (32, 32, 64, 64), 
              help = "Specify conv layer sizes, <32, 32, 64, 64> by default")
def main(epoch, rate, decay, continue_at, batch_size, channels):
    base_learning_rate = rate;
    decay_schedule = nn.decay_schedule(epoch, decay);
    learning_rate = base_learning_rate * decay_schedule;
    if(continue_at != None and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model(*channels);

    acc_curve = plot.plot();
    loss_curve = plot.plot();
    for ep in range(epoch):
        lr = learning_rate[ep];
        train = data.preprocess_training_set();
        print("training set(%d instances) loaded and shuffled" % (len(train)));
        X, Y = data.sample_batches_train(train, batch_size);
        del train;
        print("epoch %d/%d, batch size %d, learning rate %g" % (ep+1, epoch, batch_size, lr));

        model['epoch'] += 1;
        yes, cnt, epoch_loss = 0, 0, 0;
        stime = time.perf_counter();
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
            acc = yes/cnt*100;
            print(" %d/%d, acc %.2f%%, batch loss %.2f   " % (yes, cnt, acc, loss), end = '\r');
            acc_curve.append(acc);
            nn.adam_update(model, lr);
            # nn.momentum_update(model, lr);
            # nn.sgd_update(model, lr);
        etime = time.perf_counter();
        print();
        print("epoch %d/%d, overall loss %.2f, time elapsed %.2f second(s)" % (ep+1, epoch, epoch_loss, etime-stime));
        data.save_model(model);
        loss_curve.append(epoch_loss);
        acc_curve.save("acc.png");
        loss_curve.save("loss.png");
        print("model saved");
        print();

if(__name__ == "__main__"):
    main();
