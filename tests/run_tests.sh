#!/bin/bash

# Run all unittests in the tests directory

export PYTHONPATH=..
python -m unittest discover -s . -p "*_test.py"
