# Visual Pollution Object Detection

------

This project aims to build a new field of automated visual pollution classification.


In this project, I trained a custom model using Tensorflow Object Detection API version 2 on Google Colab to detect and classify 10 types of visual pollution on street imagery taken from a moving vehicle in Saudi Arabia.


## Visual pollution types:

1- BAD BILLBOARD 

2- BROKEN_SIGNAGE 

3- CLUTTER_SIDEWALK 

4- CONSTRUCTION ROAD 

5- FADED SIGNAGE 

6- GARBAGE

7- GRAFFITI

8- POTHOLES 

9- SAND ON ROAD

10- UNKEPT_FACADE


## Technology and tools used:

* Google Colab

* Python 3.8.10

* TensorFlow 2.11.0

* Tensorflow 2 object detection API


# Project steps/pipline:

* ## 1- Building the dataset (Data preparation)

I used a dataset collected and created by SDAIA (at Smartathon Hackathon). The dataset features the raw sensor camera inputs as perceived by a fleet of multiple vehicles in a restricted geographic area in KSA.
In this project, I took a subset(700 images) from the large dataset, then divided them into 500 images to feed into the model for the training phase and 200 images for the testing phase.

* ## 2- Annotating the dataset 

CSV annotation files (train.csv and test.csv) were delivered by SDAIA (at Hackathon). Although I didn't need to do the annotations myself, there was a problem with the coordinates of the bounding boxes where they weren't capturing the detected objects correctly, so I had to correct the bounding box annotations by multiplying them by 2. ([in this Pre-process the dataset jupyter notebook](https://github.com/kalthommusa/Visual-Pollution-Object-Detection/tree/master/preprocessing-dataset))
