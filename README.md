# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Build a Behavioral Cloning Network

### Overview

In this project, we will use what we've learned about deep neural networks and convolutional neural networks to build a behavioral network. we drive the car around the track in the simulator, collect the data, and then train a deep neural network to do the driving for us!

### Final Result

The final model can drive the car succesfully in track 1. (https://youtu.be/JmJUB54CtmU)

### Model Architecture
This project basically use the architecture by Nvidia paper (https://review.udacity.com/#!/rubrics/432/view). Specifically it has below layers:

1. input layer, 3@80x160
2. Normalization layer, 3@80x160
3. Conv layer, 24 filter size 5*5 + batchnormaization + relu, valid padding
4. Conv layer, 36 filter size 5*5 + batchnormaization + relu, valid padding
5. Conv layer, 48 filter size 5*5 + batchnormaization + relu+dropout, valid padding
6. Conv layer, 64 filter size 3*3 + batchnormaization + relu, valid padding
7. Conv layer, 64 filter size 3*3 + batchnormaization + relu + dropout, valid padding
8. Fully connected layer 1164+ batchnormaization + relu + dropout
9. Fully connected layer 100+ batchnormaization + relu + dropout
10. Fully connected layer 50+ batchnormaization + relu + dropout
11. Fully connected layer 10+ batchnormaization + relu + dropout
12. output layer 1


### Training Strategy

*Training data collection  
Training data collection is the most critical aspect determining the success/failure of the model in this project.
  - drive a couple of laps of nice, centerline driving
  - drive to the left of the road, and re-center, but only take recorded images that do the recovery (steering angel being positive)
  - drive to the right of the road, and re-center, but only take recorded images that do the recovery (steering angel being negative)
  - use left/right side camera images (getting their steering angle label by referecing that of center images)
  - At the end, we have 128,958 training samples availalbe
  - Remove most of the images whose steering angle is near to zero, increase appropriately the proportion of left/right turn images in the final trainig samples (their steering angle being 0.3-0.8 or -0.3-0.8).
  - At the end, we select 75,936 images as final trainig samples 



*Train/validation/test splits  
the first lap (6000) serve as validation dataset, the rest as training data. the model is eventualy tested in the simulator

*Regularization methods  
Dropout is used in an attempt to regularize the model. On the other side, the model is bound to overfit since the valdiation dataset (the first lap of driving) comes from the same track as the training dataset. To build a model that generalize well, we will need to add driving images from different lighting/evnironment.

*Optimization methods  
Default Adam optimiser from Keras package is used.


### Implementation Details

*Data preparation  
Image path and labels loading, train/val split is implemented by PrepareData class in myimagedatagenerator file
* Data selection  
Selecting the right proportion of training images from available images is implemented by DataSelection class in file dataselection  
* Image generator  
An image generator is developed to allow us to generate batches of images and feed into network on the fly , without having to store all the images in memory in advance. It's implemented by MyImageDataGenerator class in myimagedatagenerator file
* Data augmentation  
flipping, shift, ratation is implemented by DataAugmentation in in myimagedatagenerator file. This trick is not used at the end since we've already got engouh images to train track 1 by driving around the track.


### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Keras](https://keras.io/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

