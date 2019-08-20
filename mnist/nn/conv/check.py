#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import data 
import numpy as np 
import nn_utils as nn
import click 
import plot 

import warnings
warnings.filterwarnings("error")

@click.command()
@click.option("--epoch", type = int, default = 1000, 
              help = "Specifies number of epoches, 1000 by default")
@click.argument("rate", type = float)
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
def main(epoch, rate, continue_at):
    batch_size = 64;
    train_size = 64;
    learning_rate = rate;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model();
    train = data.preprocess_training_set()[0:train_size];
    loss_curve = plot.plot();
    for ep in range(epoch):
        lr = learning_rate;
        np.random.shuffle(train);
        X, Y = data.sample_batches_train(train, batch_size);
        yes, cnt, epoch_loss = 0, 0, 0;
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
            nn.update(model, lr);
        loss_curve.append(epoch_loss);
        loss_curve.save("loss.png")
        print("ep %d/%d, acc %0.2f%%, overall loss %.2f" % (ep+1, epoch, yes / cnt * 100, epoch_loss));
        # data.save_model(model);

if(__name__ == "__main__"):
    main();
