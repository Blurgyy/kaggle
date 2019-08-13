#!/usr/bin/python3

import os 
import numpy as np 
import data 
import nn 
import plot 
import click 

@click.command()
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
def main(continue_at):
    epoch = 10;
    learning_rate = 1e-3;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model();

    yes = 0;
    cnt = 0;
    for ep in range(epoch):
        train = data.preprocess_training_set();
        print("training set loaded and shuffled");
        for elem in train:
            label, img = elem;
            img = np.array([img.reshape(28, 28)]);
            nn.forward(model, img, is_test_time = False);
            prob = model['output'].copy();
            prob -= np.max(prob);
            prob = np.exp(prob) / np.sum(np.exp(prob));
            dz = prob.copy();
            dz[label] -= 1;
            nn.backward(model, dz);
            nn.update(model, learning_rate);

            predict = np.argmax(model['output']);
            yes += (predict == label);
            cnt += 1;
            if(cnt % 1 == 0):
                # loss_curve.append(-np.log(prob[label]));
                print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100));
                data.save_model(model);

if(__name__ == "__main__"):
    main();