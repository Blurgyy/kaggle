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
    learning_rate = 1e-1;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model();
    train = data.preprocess_training_set();
    dat = [train[0], train[1], train[2]];
    for ep in range(epoch):
        random.shuffle(dat);
        for elem in dat:
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
            loss = -np.log(prob[label]);
            pred = np.argmax(model['output']);
            print("epoch %d, loss %g, ground truth [%d], prediction [%d], p/gt = %.2f%%/%.2f%%" % (ep, loss, label, pred, prob[pred]*100, prob[label]*100));
        print();
        data.save_model(model);

if(__name__ == "__main__"):
    main();
