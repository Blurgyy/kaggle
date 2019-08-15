#!/usr/bin/python

import data 
import numpy as np 
import nn
import click 
import random 

@click.command()
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
def main(continue_at):
    epoch = 1000;
    batch_size = 5;
    learning_rate = 1e-2;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model();
    train = data.preprocess_training_set()[0:10];
    yes = 0;
    cnt = 0;
    loss = 0;
    for ep in range(epoch):
        lr = learning_rate;
        random.shuffle(train);
        X, Y = nn.sample_batches(train, batch_size);
        for i in range(len(X)):
            x, y = X[i], Y[i];
            nn.forward(model, x, is_test_time = False);
            prob = model['output'].copy();
            prob -= np.max(prob, axis=1).reshape(-1,1,1);
            prob = np.exp(prob) / np.sum(np.exp(prob), axis=1).reshape(-1,1,1);
            dz = prob.copy();
            a0 = np.arange(batch_size).reshape(-1,1);
            y = y.reshape(-1,1);
            a2 = np.repeat(0, batch_size).reshape(-1,1);
            dz[a0,y,a2] -= 1;
            nn.backward(model, dz);

            prediction = np.argmax(model['output'], axis=1).reshape(-1,1);
            yes += np.sum(prediction == y);
            cnt += batch_size;
            if(cnt % batch_size == 0):
                print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100), end = '\r');
            nn.update(model, lr);
        data.save_model(model);

if(__name__ == "__main__"):
    main();
