# semi_supervised_mnist 
This is the group project for in-class Kaggle competition [MNIST_DIGIT_RECOGNITION_ASSIGNMENT_1](https://inclass.kaggle.com/c/mnist-digit-recognition-assignment-1). The class is DS-GA 1008 Deep Learning at NYU, provided by Yann Lecun. 

The problem setting for this competition is Semi-Supervise Learning. We have 3,000 labeled samples and 47,000 unlabeled sampples. Using ConvNets, along with data augmentation, pseudo-label method, and ensembling, we get a 99.42% accuracy on 10,000 test samples, ranking the 2nd out of 30 teams.

## 1. General Methods
### ConvNets Architecture
The input to our ConvNets is a fixed-size 28![](http://latex.codecogs.com/gif.latex?*)28 single channel images of hand-written numbers. The image is passed through a stack of two convolutional layers of 5![](http://latex.codecogs.com/gif.latex?*)5 kernels. The convolution stride is fixed to 1 pixel. Spatial pooling is carried out by five max-pooling layers, which follow the conv layers. Max-pooling is performed over a 2![](http://latex.codecogs.com/gif.latex?*)2 pixel window, with stride 1.

### Data Augmentaion
We performed rotate, skew, affine, and randomcrop transformation to 3,000 labeled samples. The reasons for doing this are avoiding overfitting and trying to mimic as many different styles of handwriting of digitals as possible. Since there are no build-in method for rotating, skew, and affine transformation on image in Pytorch so far, we add those three methods to torchvision.transforms class. So in order to run our code, you may want to update torchvision.transforms.py file with the one we provide.

### Pseudo-Label
We use Pseudo-Label method to incorporate information from unlabeled samples, which works as the following three steps:

1. Train model on 15,000 labeled data(3,000 original plus 12,000 augmented)

In this supervised learning step, we trained 20 epochs. This phase is to stabilize our classifier to predict a relatively reliable label for unlabeled data

2. Gradually using information of unlabeled data

We first used the model from last epoch to provide them pseudo labels, which are recalculated every weights update. Since we add unlabeled data, the overall loss function becomes:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![](http://latex.codecogs.com/gif.latex?L%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bm%3D1%7D%5E%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7BC%7DL%28y_i%5Em%2Cf_i%5Em%29&plus;%5Calpha%28t%29%5Cfrac%7B1%7D%7Bn%27%7D%5Csum_%7Bm%3D1%7D%5E%7Bn%27%7D%5Csum_%7Bi%3D1%7D%5E%7BC%7DL%28y%27%7B_i%5Em%7D%2Cf%27%7B_i%5Em%7D%29)

![](http://latex.codecogs.com/gif.latex?%5Calpha) controls how much emphasis we put on unlabeled samples, and in this step, ![](http://latex.codecogs.com/gif.latex?%5Calpha) is growing from 0 to 3.1 during the training process, as we become more and more confident on the pseudo labels.
Another trick we found useful to improve the final model accuracy is let the training epoch loop more on the labeled data to serve as a self correction. In our final training process, during every epoch of looping through all the unlabeled data, we loop through the labeled data for 7 times. This phase lasts for 20 epochs.

3. Train model on both unlabeld and labeled data with fixed ![](http://latex.codecogs.com/gif.latex?%5Calpha)

During the final phase, we did exactly the same as in phase 2, except that we fixed ![](http://latex.codecogs.com/gif.latex?%5Calpha) to be 3.1. This is the best value we get from the fine tuning. By monitoring the validation error, the entire process converged after 100 epoch.

### Ensembling
To reduce the variance and increase accuracy of our model, we trained 10 different models with different hyper-parameters on 10 boostrap samples separately, and took the average of their outputs.

## 2.running our code
### Preparation
Code in this repo should be run under environment Python 2.7.0 or higher.
Use 
```bash
pip install -r requirements.txt
```
to install all the dependencies to run the code in this repo. 

To unzip the data being used, please go to the /data directory and run
```bash
unzip data.zip
```
### Model Training
Change to the /code directory and run 
```bash
python mnist_model.py 
```
This will train the model using all default parameters, which can generate a similar result we mentioned. The trained model will be saved under the /model directory. For futher parameter tuning, you can check the argument parser in the code to setup your choices. 
### Prediction Generating
Under the /code directory, you can run 
```bash
python mnist_result.py
```
This will generate prediction output for each of the test point and the result should be saved as submision.csv.
