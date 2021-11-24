#!/bin/bash
for filename in data/test/image/*; do
  clipped_name=${filename##*/}
  python predict.py -i data/test/image/$clipped_name -o data/test/mask/${clipped_name%.*}_mask.png --model checkpoints/checkpoint_epoch5.pth
done
