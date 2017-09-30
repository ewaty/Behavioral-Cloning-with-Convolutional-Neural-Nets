# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
    * Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.jpg "Center driving"
[image2]: ./examples/image2.jpg "Recovery Image"
[image3]: ./examples/image3.jpg "Recovery Image"
[image4]: ./examples/image4.jpg "Recovery Image"
[image5]: ./examples/image5.jpg "Flipped Image"
[image6]: ./examples/image6.jpg "Flipped Image"
[image7]: ./examples/image7.jpg "Left Image"
[image8]: ./examples/image8.jpg "Center Image"
[image9]: ./examples/image9.jpg "Right Image"

---
### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.html summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture that has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths 6, 18 and 54 (model.py lines 78-83) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 76). Images are cropped to remove parts containing unnescessary elements like dashboard, trees and sky (code line 77).

A convolutional neural network is followed by a fully connected network with three layers, including output (code lines 85-87). The fully connected layers are interlaced with a softsign activation function

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model returned similar train and validation loss therefore it can be said thet it is not overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to modify an existing neural network.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because I have used it before on a Udacity project about classifying traffic signs. I have made the network leaner by shortening convolutional layers. This has turned out not to have a negative or positive effect on validation accuracy.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I changed the activation functions of the fully connected layers from ReLu to SoftMax. This helped improve the validation accuracy.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected data when the vehicle was correcting its trajectory creating a recovery set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-87) consisted of a convolution neural network with the following layers and layer sizes:
* input layer of size (160, 320, 3) - cropped to (65, 320, 3)
* convolutional layer of depth 6
* ReLu activation
* max pooling
* convolutional layer of depth 18
* ReLu activation
* max pooling
* convolutional layer of depth 54
* ReLu activation
* max pooling
* fully connected layer of size 108
* SoftSign activation function
* fully connected layer of size 54
* SoftSign activation function
* output layer of size 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded six laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its trajectory when it has failed to remain in middle. These images show what a recovery looks like starting from the car being pointed towards the right side of the road and then turning left to return to the middle:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data set, I also flipped images and angles thinking that this would create more data and help the network generalize better. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

I also used images from left and right cameras for training. To correct for their different perspectives I added bias of 0.2 for left camera and -0.2 for right camera. Below I show images from the same timestamp for left center and right cameras.

![alt text][image7]
![alt text][image8]
![alt text][image9]

After the collection process, I had 37257 data points. I then preprocessed this data by normalizing it and cropping out unnescessary parts of image like the dashboard and trees.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the graph of validation error I per each epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The results of the network training can be seen in the "run1.mp4" video. The car is able to clear the entire track.
