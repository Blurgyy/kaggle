#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import data 
import numpy as np 
import nn_utils as nn
import click 
import time 
import plot 

import warnings
warnings.filterwarnings("error")

@click.command()
@click.option("--epoch", type = int, default = 1000, 
              help = "Specifies number of epoches, 1000 by default")
@click.argument("rate", type = float)
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
@click.option("--batch-size", type = int, default = 64, 
              help = "Specifies batch size, 64 by default")
def main(epoch, rate, continue_at, batch_size):
    train_size = 64;
    learning_rate = rate;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model(4, 16, 16, 16);
    train = data.preprocess_training_set()[0:train_size];
    loss_curve = plot.plot();
    acc_curve = plot.plot();
    for ep in range(epoch):
        lr = learning_rate;
        np.random.shuffle(train);
        X, Y = data.sample_batches_train(train, batch_size);
        yes, cnt, epoch_loss = 0, 0, 0;
        stime = time.perf_counter();
        for i in range(len(X)):
            x, y = X[i], Y[i];
            nn.forward(model, x, is_test_time = False);
            dz, loss = nn.grad(model, y);
            nn.backward(model, dz);

            epoch_loss += loss;
            prediction = np.argmax(model['output'], axis=1);
            score = prediction.reshape(-1,1) == y.reshape(-1,1);
            yes += np.sum(score);
            cnt += len(y);
            nn.adam_update(model, lr);
            # nn.momentum_update(model, lr);
            # nn.sgd_update(model, lr);
        etime = time.perf_counter();
        acc = yes/cnt*100;
        loss_curve.append(epoch_loss);
        acc_curve.append(acc);
        loss_curve.save("loss.png")
        acc_curve.save("acc.png");
        print("ep %d/%d, acc %0.2f%%, loss %.2f, time elapsed %.2f second(s)" % (ep+1, epoch, acc, epoch_loss, etime-stime));
        # data.save_model(model);

if(__name__ == "__main__"):
    main();
