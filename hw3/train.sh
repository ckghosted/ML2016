#!/bin/bash
THEANO_FLAGS=device=gpu0 python autoencoder.py $1 $2
