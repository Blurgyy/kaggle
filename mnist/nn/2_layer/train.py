#!/usr/bin/python3 -u 
# -*- coding: utf-8 -*-
__author__ = "Blurgy";

import os 
import numpy as np 
import data 
import nn 
import click 
import plot 

@click.command()
@click.option("--epoch", type = int, default = 10, 
              help = "Specifies number of epoches, 10 by default")
@click.option("--rate", type = float, default = 1e-3, 
              help = "Specifies value of initial learning rate, 1e-3 by default")
@click.option("--reg", type = float, default = 1e-6,
              help = "Specifies regularization strenth, 1e-6 by default")
@click.option("--decay", type = click.Choice(["exponential", "constant", "linear", "sigmoid", "hyperbola"]), 
              default = "exponential", 
              help = "Specifies decay schedule of learning rate, exponential by default")
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
@click.option("--batch-size", type = int, default = 10,
              help = "Specifies batch size, 10 by default")
def main(epoch, rate, reg, decay, continue_at, batch_size):
    input_size = 784;
    hidden_layer_size = 200;
    output_size = 10;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model(input_size, hidden_layer_size, output_size, reg);

    # epoch = 20;
    # base_learning_rate = 1e-5;
    base_learning_rate = rate;
    decay_schedule = nn.decay_schedule(epoch, decay);
    learning_rate = base_learning_rate * decay_schedule;
    print(learning_rate);

    precision_curve = plot.plot();
    loss_curve = plot.plot();
    for ep in range(epoch):
        lr = learning_rate[ep];
        training_set = data.preprocess_training_set();
        print("training epoch %d/%d with learning rate %g" % (ep+1, epoch, lr));
        batches = nn.sample_batches(training_set, batch_size);
        yes = 0;
        cnt = 0;
        for batch in batches:
            for item in batch:
                label, img = item;
                nn.forward(model, img, is_test_time = False);
                prob = model['score'].copy();
                prob -= np.max(prob);
                prob = np.exp(prob) / np.sum(np.exp(prob));
                dz = prob.copy();
                dz[label] -= 1;
                dz /= batch_size;
                # nn.sgd_backward(model, dz, lr);
                nn.adam_backward(model, dz, lr);

                predict = np.argmax(model['score']);
                yes += (predict == label);
                cnt += 1;
                if(cnt % 1000 == 0):
                    loss_curve.append(-np.log(prob[label]));
                    print("[%d/%d]: %0.2f%%" % (yes, cnt, yes / cnt * 100), end = '\r');
            # dz /= batch_size;
        precision_curve.append(yes/cnt);
        precision_curve.save("precision.jpg");
        loss_curve.save("loss.jpg");
        data.save_model(model);
        print("\nmodel saved\n");


if(__name__ == "__main__"):
    main();
