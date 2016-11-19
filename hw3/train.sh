#!/bin/bash
THEANO_FLAGS=device=gpu0 python self_training.py $1 $2
