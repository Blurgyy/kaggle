#!/usr/bin/python3

import os 
import numpy as np 
import data 
import nn 
import plot 
import click 

@click.command()
@click.option("--epoch", type = int, default = 10, 
              help = "Specifies number of epoches, 10 by default")
@click.option("--rate", type = float, default = 1e-2, 
              help = "Specifies value of initial learning rate, 1e-2 by default")
@click.option("--decay", type = click.Choice(["exponential", "constant", "linear", "sigmoid", "hyperbola"]), 
              default = "exponential", 
              help = "Specifies decay schedule of learning rate, exponential by default")
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
@click.option("--batch-size", type = int, default = 10,
              help = "Specifies batch size, 10 by default")
def main(epoch, rate, decay, continue_at, batch_size):
    base_learning_rate = rate;
    decay_schedule = nn.decay_schedule(epoch, decay);
    learning_rate = base_learning_rate * decay_schedule;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model();

    yes = 0;
    cnt = 0;
    for ep in range(epoch):
        lr = learning_rate[ep];
        train = data.preprocess_training_set();
        print("training set loaded and shuffled");
        batches = nn.sample_batches(train, batch_size);
        for batch in batches:
            for elem in batch:
                label, img = elem;
                img = img.reshape(1, 28, 28);
                nn.forward(model, img, is_test_time = False);
                prob = model['output'].copy();
                prob -= np.max(prob);
                prob = np.exp(prob) / np.sum(np.exp(prob));
                dz = prob.copy();
                dz[label] -= 1;
                nn.backward(model, dz);

                pred = np.argmax(model['output']);
                yes += (pred == label);
                cnt += 1;
                if(cnt % 1 == 0):
                    print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100), end = '\r');
            nn.update(model, lr/batch_size);
            data.save_model(model);
            print("\nupdated", end = '\r');

if(__name__ == "__main__"):
    main();