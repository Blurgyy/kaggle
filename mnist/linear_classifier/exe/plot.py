#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import sys
import numpy as np
from matplotlib import pyplot as plt

def plot(fname, ):
    fn_name = "plot";
    try:
        lines = None;
        xaxis = []; # epoch id
        yaxis = []; # precision
        lraxis = []; # learning rate
        with open(fname) as f:
            lines = f.readlines();
        epoch = len(lines);
        for i in range(epoch):
            prec = float(lines[i].strip().split(' ')[0]);
            lr = float(lines[i].strip().split(' ')[1]);
            xaxis.append(i+1);
            yaxis.append(prec);
            lraxis.append(lr);
        fig = plt.figure();
        y1 = fig.add_subplot(111);
        y1.plot(xaxis, yaxis, label = "precision");
        y1.axis([1, xaxis[-1], yaxis[0], 1.0]);
        y1.legend(loc = 3);
        plt.grid();

        y2 = y1.twinx();
        y2.tick_params(axis = 'y', colors = "orange");
        y2.plot(xaxis, lraxis, "orange", label = "learning rate");
        y2.legend(loc = 1);
        plt.grid();
        plt.savefig(fname.split('.')[-2] + '.png');
        plt.show();
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        fname = ".training_precision.log";
        if(len(sys.argv) == 2):
            fname = sys.argv[1];
        plot(fname);
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

if(__name__ == "__main__"):
    main();
