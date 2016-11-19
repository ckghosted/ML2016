#!/bin/bash
THEANO_FLAGS=device=gpu0 python self_training_test.py $1 $2 $3
