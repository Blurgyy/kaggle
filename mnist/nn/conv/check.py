#!/usr/bin/python

import data 
import numpy as np 
import nn_utils as nn
import click 
import random 

@click.command()
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
def main(continue_at):
    epoch = 1000;
    batch_size = 10;
    train_size = 20
    learning_rate = 1e-4;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model(1e-3);
    train = data.preprocess_training_set()[0:train_size];
    for ep in range(epoch):
        lr = learning_rate;
        random.shuffle(train);
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
        print("ep %d/%d, acc %0.2f%%, overall loss %.2f" % (ep+1, epoch, yes / cnt * 100, epoch_loss));
        # data.save_model(model);

if(__name__ == "__main__"):
    main();
