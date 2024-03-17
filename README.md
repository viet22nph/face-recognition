
# Face Recognition with OpenCV and Python

## Coding Face Recognition with OpenCV
The Face Recognition process in this tutorial is divided into three steps.

1. **Prepare training data:** In this step we will read training images for each person/subject along with their labels, detect faces from each image and assign each detected face an integer label of the person it belongs to.
2. **Train Face Recognizer:** In this step we will train OpenCV's LBPH face recognizer by feeding it the data we prepared in step 1.
3. **Testing:** In this step we will pass some test images to face recognizer and see if it predicts them correctly.

**[There should be a visualization diagram for above steps here]**

To detect faces, I will use the code from my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/). So if you have not read it, I encourage you to do so to understand how face detection works and its Python coding. 


### Import Required Modules
### Import Required Modules

Before starting the actual coding we need to import the required modules for coding. So let's import them first. 

- **cv2:** is _OpenCV_ module for Python which we will use for face detection and face recognition.
- **os:** We will use this Python module to read our training directories and file names.
- **numpy:** We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.

```python
import cv2
import os
import numpy as np

```
### Create data 
Before training, we need to create a dataset. To do this, we need to read the videos and save the images to the datasets folder. 
Run the create_data file to generate the dataset so that the training can be done.
### Training Data
The more images used in training the better. Normally a lot of images are used for training a face recognizer so that it can learn different looks of the same person, for example with glasses, without glasses, laughing, sad, happy, crying, with beard, without beard etc. To keep our tutorial simple we are going to use only 110 images for each person. 

The _`test-data`_ folder contains images that we will use to test our face recognizer after it has been successfully trained.

As OpenCV face recognizer accepts labels as integers so we need to define a mapping between integer labels and persons actual names so below I am defining a mapping of persons integer labels and their respective names. 

### Test data
After training the data, we perform tests with all non-identical images in the dataset generated during data generation. The purpose of performing the test to see the correct percentage with each trained face helps to see what percentage of training is achieved
## End Notes

Face Recognition is a fascinating idea to work on and OpenCV has made it extremely simple and easy for us to code it. It just takes a few lines of code to have a fully working face recognition application and we can switch between all three face recognizers with a single line of code change. It's that simple. 

Although EigenFaces, FisherFaces and LBPH face recognizers are good but there are even better ways to perform face recognition like using Histogram of Oriented Gradients (HOGs) and Neural Networks. So the more advanced face recognition algorithms are now a days implemented using a combination of OpenCV and Machine learning. I have plans to write some articles on those more advanced methods as well, so stay tuned! 
