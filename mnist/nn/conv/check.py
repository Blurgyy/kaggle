#!/usr/bin/python

import data 
import numpy as np 
import nn
import click 

@click.command()
@click.option("--continue-at", type = click.Path(exists=True), default = None,
              help = "Continues training at specified file, initializes a new model if not specified")
def main(continue_at):
    epoch = 1000;
    learning_rate = 1e-2;
    if(continue_at and os.path.exists(continue_at)):
        model = data.load_model(continue_at);
    else:
        model = nn.init_model();
    elem = data.preprocess_training_set()[0];
    print(elem[0]);
    for ep in range(epoch):
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
        print("epoch %d, loss %g, ground truth [%d], prediction [%d]" % (ep, loss, label, np.argmax(model['output'])));
        data.save_model(model);

if(__name__ == "__main__"):
    main();
