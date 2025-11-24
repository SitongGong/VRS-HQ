#!/usr/bin/env bash
source ~/miniconda3/bin/activate freeva
conda env list
cd /18515601223/segment-anything-2/tools
python eval_davis17.py