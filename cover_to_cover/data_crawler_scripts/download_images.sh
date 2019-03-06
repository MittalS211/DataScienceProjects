#!/bin/bash
set -eu


output_dirpath="Desktop/images"
csv_filepath="book30_listing_train.csv"
python3 download_images.py ${output_dirpath} ${csv_filepath}
