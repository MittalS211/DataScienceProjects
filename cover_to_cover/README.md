This Repository contains all files for Cover to Cover, a content based Image Retrieval recommender system  of the Amazon book covers dataset with similarity algorithms in Computer Vision.

The project files

Notebooks:

**NOTE** if Notebooks are not rendering please copy the url go to https://nbviewer.jupyter.org/ to view the files

Cover_to_cover.ipynb contains the Similarity based Image Recommendation System

genre_classification.ipynb explores the relationship between genre and cover design features with Neural network modeling


Directories:

models/ contains all models used for the Project

model_history/ contains results from ALL tests on data set

modeling_pipeline/ contains code for neural network pipeline and plotting function used in genre_classification, the notebook contains full documentation of pipeline for reference purposes. 

data_crawler_scripts/ contains modified python and shell scripts for retrieving book cover images from  https://github.com/uchidalab/book-dataset and use these scripts to bypass errors from missing/broken image urls
