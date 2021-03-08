# Machine Learning with CNN

A machine learning project in Python.
The project includes both training and testing GUI.
The aim of this project is (English) character recognition.

## The training GUI
This simple GUI allows the user to make a machine learning model without even typing any code.
The structure of the program allows easy expansion of it, that is, adding more layers, optimizers etc. to the program
is not a hassle. The training program currently only allows the usage of images as training data, given that these images
are already sorted in test/train folders. Further documentation is yet to be written, but generally speaking, the
program is easy to use and yet offers a decent amount of functionality.
### How to use the program
#### Selecting the data
The usage of the program is simple:
1. Select the training and the test folder by clicking on the **Train folder** and the **Test folder** buttons and choose the correct folders.
2. Select the model save location with the **Save folder** button.
#### Building the model
To start building the model, use the interface on the left:
1. Enter the training set size next to the **Train Size** button. Clicking on the **Train Size** button sets the train size to auto, meaning all the images are used in the training process.
2. Enter the test set size next to the **Test Size** button. Clicking on the **Test Size** button sets the test size to auto, meaning all the images are used in the training process.
3. Enter the number of classes next to the **Classes** button. Clicking on the **Classes** button sets the number of classes to auto, identifying the number of classes automatically before the training process.
4. Enter the image size next to the **Image Size** button. Clicking on the **Image Size** button sets the image size to a default value of 64 by 64 pixels.
5. Enter the batch size below the **Batch Size** button. Clicking on the **Batch Size** button sets the batch size to a default value of 32.
6. Enter the number of epochs below the **Epochs** button. Clicking on the **Epochs** button sets the number of epochs to a default value of 10.
7. Select the model optimizer below the **Optimizers** field:
   * Adam,
   * SGD.
   [//]: # (New line)
   The selected optimizer is also displayed in the widget on the right part of the interface below the **Optimizer** field.
8. Enter the learning rate below the **LR** button. Clicking on the **LR** buttpn sets the learning rate to a deault value of 0,01.
9. Select the loss function below the **Loss function** field by clicking on the desired loss function:
   * Binary CE (Binary Cross-Entropy),
   * Categorical CE (Categorical Cross-Entropy),
   * MSE (Mean squared error).
   The selected loss function is also displayed in the widget on the right part of the interface below the **Loss function** field.
10. Select the metrics below the **Metrics** field by clicking on the desired metrics:
   * ACC (Accuracy),
   * MAE (Mean Absolute Error).
   The selected metrics is also displayed in the widget on the right part of the interface below the **Metrics** field.
11. Now start building the model by adding the layers located below the **Add Layer** button. First select the desired layer and by clicking on the **Add Layer** button add the selected layer to the model. The most important available layers are:
   * Input (Input layer, not neccessary, a convolution layer can be used instead),
   * Convolution2D (Convolutional 2D layer),
   * MaxPooling2D (Max Pooling 2D layer),
   * Dropout (Dropout layer),
   * Flatten (Flatten layer).
##### Setting the layer parameters
Each layer has its corresponding properties, which can be set inside of the layer itself. All of the values in the layer(s) have to be set or the model won't compile and run. Layer properties need to be entered inside of the corresponding text fields equiped with the property label name. To remove a specific layer, click on the **Remove** button located inside of the layer itself.
#### Post and pre model-building
##### Getting the final (output) shape
Below the **Output shape** field are displayed the final output layer shape (after and before flatten). To obtain the following information, first build or load the model and after that use the **Get shape** button, to get the output shapes.
##### Saving and loading the model
After a model has been built, its structure can be saved into a .nets file. To do so, after the model is built, click on the **Save network** button and select the save location and the file save name.
Models can also be loaded from these same files by using the **Load network** button. After clicking the **Load network** button, select the network that you want to load. Clicking on select in the explorer window will result into the old model being deleted and the new model being loaded into the program.
 

## The test GUI
This program offers an easy way to test the model performance. First the user selects a model and after that an
image. The last thing that needs to be done after this, is clicking the predict button, which, of course, gives back
the prediction made by the model. The program was adapted to work only with 62 characters dataset. Further improvements
might be made in the near future.

## The CNN model
The CNN model was made using the training GUI and the best model I managed to train so far reaches 86 % test accuracy.
The main reason for this average score is the dataset I used since using other datasets I managed to achieve much
better accuracy using the same model. Another reason for the average performance of the model is the large amount of
classes, since this model was trained using 62 different characters.
Better performance might be achieved by increasing the number of layers in the network.
#### Update 
I managed to create a better model (91% val acc). Explanation below:
##### The 91% Model
#TODO

## The (old) backend
The backend in this project is not Tensorflow/ Tensorflow GPU as usually. Instead, the backend used in this project is
PlaidML: https://github.com/plaidml/plaidml. The main reason for usage of this backend is the lack of an Nvidia GPU in 
my computer, since Tensorflow GPU works only with GPUs that support CUDA core technology (that means Nvidia GPUs only).
PlaidML relies on OpenGL instead, allowing the usage of any GPU (Amd GPU in my case) to speed up the model learning
process. A GPU can speed up the learning process by at least 50%.
#### Update
Later in the project the backend was updated from PlaidML to Tensorflow GPU due to harware change in my personal computer.
Basically I changed from an AMD (RX 480) GPU, which does not support CUDA to an Nvidia (RTX 3080) GPU, which does support
CUDA and thus I was able to use Tensorflow GPU which is based on the CUDA technology.

## Future plans
The idea is to improve the model performance and expand the training/testing program functionality.

## This project's resources
* Training and test dataset: 
https://catalog.data.gov/dataset/nist-handprinted-forms-and-characters-nist-special-database-19.
* Tkinter library information: https://docs.python.org/3/library/tkinter.html, 
https://www.tutorialspoint.com/python/python_gui_programming.htm.
* Everything about machine learning: https://www.kaggle.com/.
* A useful Kaggle guide: https://www.kaggle.com/poonaml/deep-neural-network-keras-way.
*  Nils J. Nilsson, 2010: <i/>[The Quest for Artificial Intelligence.](https://www.goodreads.com/book/show/7465939-the-quest-for-artificial-intelligence "Good reads")</i>
* Tariq Rashid, 2016: <i/>[Make Your Own Neural Network: A gentle journey through the mathematics of neural networks,
and marking your own using the Python computer language.](https://books.google.si/books/about/Make_Your_Own_Neural_Network.html?id=Zli_jwEACAAJ&source=kp_book_description&redir_esc=y "Google books")</i>
* C.-C. Jay Kuo, 2016: <i/>[Understanding Convolutional Neural Networks with A Mathematical Model.](https://arxiv.org/abs/1609.04112 "Arxiv")</i>
* Max Tegmark, 2017: <i/>[Life 3.0: Being Human in the Age of Artificial Intelligence.](https://www.goodreads.com/book/show/34272565-life-3-0 "Good reads")</i>

## Required Python libraries
* datetime
* os
* tkinter
* ~~plaidml (Link: https://github.com/plaidml/plaidml)~~
* keras
* Tensorflow GPU...

## Other requirements
* CUDA GPU (Nvidia)
* cudNN
* Python

### TODO LIST
~~Run the training in a separate process, solving the memory problem.~~
~~Optimize the code (method generalization, removal of obsolete code).~~
~~Update the old documentation and the word document with proper language and the changes in the project
(new code, new features, different data processing etc.).~~
