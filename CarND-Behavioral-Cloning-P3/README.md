# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### **Submission Note**: Please find `model.py`, `preprocess.py`, and `video.mp4`.


## Rubric Points
--------------------------------------

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results (this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My first try was Nvidia Autonomous Car Grous model which can be find in `model.py`. The results were so good that I didn't need to make any change. I was planning to train it for 5 epochs, but right after I started training I realized the loss is decreasing at good rate. So, 5 epochs are not necessary. Then, I reduced it to 3 epochs.

The model architecture is as follows:

|        Layer (type)        |     Output Shape      | Param     |
|:--------------------------:|:---------------------:|:---------:|
|lambda_1 (Lambda)           | (None, 160, 320, 3)   |    0      |    
|cropping2d_1_1 (Cropping2D))| (None, 90, 320, 3)    |    0      |    
|conv2d_1 (Conv2D)           | (None, 43, 158, 24)   |    1824   |    
|conv2d_2 (Conv2D)           | (None, 20, 77, 36)    |    21636  |    
|conv2d_3 (Conv2D)           | (None, 8, 37, 48)     |    43248  |    
|conv2d_4 (Conv2D)           | (None, 6, 35, 64)     |    27712  |    
|conv2d_5 (Conv2D)           | (None, 4, 33, 64)     |    36928  |    
|flatten_1 (Flatten)         | (None, 8448           |    0      |    
|dense_1 (Dense)             | (None, 100)           |    844900 |
|dense_2 (Dense)             | (None, 50)            |    5050   |   
|dense_3 (Dense)             | (None, 10)            |    510    |  
|dense_4 (Dense)             | (None, 1)             |    11     |  



#### 2. Attempts to reduce overfitting in the model

During training, I was monitoring my model by looking at logs. I didn't notice too much overfitting. so I didn't give Dropout or BatchNormalization a try. Also, if we check the video of autonomous driving, it seems the model has learnt very well. It almost always drives in the center. Also, I didn't have many epochs so the chances of overfitting will be reduced. 

#### 3. Model parameter tuning

The model used an adam optimizer, which has adaptive learning, so the learning rate was not tuned manually. (model.py line 33)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and reverse driving. I also augmented images by flipping them.

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

After observing the complexity of the problem and based on my experience, I decided to start with Nvidia Models which was mentioned in the lecture. I also added Lambda layer for normalization and Cropping layer to crop out only interesting part (i.e road) from the image. I trained for 3 epochs and it have me very good results. Very good that it never missed the center of the lane except once during turn. Even in that situation it got back to the center immediately. So, clearly this result was satisfactory and I didnt have to try another model.
I believe the data played a pivotal role in training. I believe I captured fairly large data with around 25000 images which consist of center lane driving, both corner driving, recovery driving and also reverse driving. I also augmented each image with flipping so technically I had 50000+ images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 8-22) is as follows:

|        Layer (type)        |     Output Shape      | Param     |
|:--------------------------:|:---------------------:|:---------:|
|lambda_1 (Lambda)           | (None, 160, 320, 3)   |    0      |    
|cropping2d_1_1 (Cropping2D))| (None, 90, 320, 3)    |    0      |    
|conv2d_1 (Conv2D)           | (None, 43, 158, 24)   |    1824   |    
|conv2d_2 (Conv2D)           | (None, 20, 77, 36)    |    21636  |    
|conv2d_3 (Conv2D)           | (None, 8, 37, 48)     |    43248  |    
|conv2d_4 (Conv2D)           | (None, 6, 35, 64)     |    27712  |    
|conv2d_5 (Conv2D)           | (None, 4, 33, 64)     |    36928  |    
|flatten_1 (Flatten)         | (None, 8448           |    0      |    
|dense_1 (Dense)             | (None, 100)           |    844900 |
|dense_2 (Dense)             | (None, 50)            |    5050   |   
|dense_3 (Dense)             | (None, 10)            |    510    |  
|dense_4 (Dense)             | (None, 1)             |    11     |  

The kernel size for convolutional layers was 3 * 3.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps of center lane driving in first track. Then, I recorded on lap of recovery driving from both sides. Then, I recorded two laps of reverse driving. This way I gathered around 25000 images.
I didn't record any data from track 2.
I also augmented my data by flipping the images. So, finally I had 50000+ images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I trained model for 3 epochs which gave me very good accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.
