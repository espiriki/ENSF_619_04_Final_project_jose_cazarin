#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-2,
                        help='regularization')
    parser.add_argument('--tl', type=bool, default=False,
                        help='transfer learning')
    parser.add_argument('--ft_epochs', type=int, default=5,
                        help='fine tuning epochs')
    parser.add_argument('--fraction_lr', type=float, default=5,
                        help='value to divide the regular LR for to use in fine tuning')

    parser.add_argument('--model', type=str, default='b4', help='model name')

    args = parser.parse_args()
    return args
