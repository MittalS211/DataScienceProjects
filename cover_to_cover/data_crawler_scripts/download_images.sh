#!/bin/bash
set -eu


output_dirpath="Desktop/train_set"
csv_filepath="book30_listing_train.csv"
python3 download_images_parallel.py ${output_dirpath} ${csv_filepath}
