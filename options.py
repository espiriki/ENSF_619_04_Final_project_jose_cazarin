#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--model', type=str, default='b4', help='model name')

    args = parser.parse_args()
    return args
