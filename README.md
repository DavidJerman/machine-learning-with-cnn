# Machine Learning with CNN

A machine learning project in Python.
The project includes both training and testing GUI.
The aim of this project is character recognition.

## The training GUI
This simple GUI allows the user to make a machine learning model without even typing any code.
The structure of the program allows easy expansion of it, that is, adding more layers, optimizers etc. to the program
is not a hassle. The training program currently only allows usage of images as training data, given that these images
are already sorted in test/train folders. Further documentation is yet to be written, but generally speaking, the
program is easy to use and yet offers a decent amount of functionality.

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
Update: I managed to create a better model (91% val acc), which is explained in detail in the wiki section of this project.

## The backend
The backend in this project is not Tensorflow/ Tensorflow GPU as usually. Instead, the backend used in this project is
PlaidML: https://github.com/plaidml/plaidml. The main reason for usage of this backend is the lack of an Nvidia GPU in 
my computer, since Tensorflow GPU works only with GPUs that support CUDA core technology (that means Nvidia GPUs only).
PlaidML relies on OpenGL instead, allowing the usage of any GPU (Amd GPU in my case) to speed up the model learning
process. A GPU can speed up the learning process by at least 50%.
### Update
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
* keras... (Read more in Wiki)
* Tensorflow GPU

## Other requirements
* CUDA GPU (Nvidia)
* cudNN
* Python

### TODO LIST
~~Run the training in a separate process, solving the memory problem.~~
~~Optimize the code (method generalization, removal of obsolete code).~~
~~Update the old documentation and the word document with proper language and the changes in the project
(new code, new features, different data processing etc.).~~
