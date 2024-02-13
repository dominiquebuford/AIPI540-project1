# FoodFables- Vegetable Classification

##Project Description
In this project, we use computer vision to classify images of vegetables, 15 classes of common vegetables. 
For our traditional ML approach, we built an SVM, which had validation and test accuracies of 81.8% and 82.8%, respectively.
For our naive approach, we used ResNet-18 without replacing and retraining the fully connected layer, and resulted in a validation score of 8.1%
For our final approach, we used Google's Inception, which had validation and test accuracies of 99.9% and 99.8%, respectively.

##Data
s3bucket_googlecolab.py: script to grab the data from our s3 bucket and save it to your Google Drive.

##Notebooks
SVC.ipynb: notebook for our SVC model development and tests

main_inception_VeggieTales_model.ipynb: notebook for training and evaluation of our final Inception model

restNetNaive.ipynb: notebook for the evaluation with resNet18, no training.

##Scripts
generateIllustration.py: script to generate an illustration using OpenAI API based on the generated story

generateStory.py: script used to call the OpenAI API to generate a child-story based on the input of detected vegetables

setup.py: script to grab data and run final Inception model.

##Models
SVC_model.py: script to run SVC model

naive_model.py: script to run naive ResNet-18 model.

main_inception_vegetable.py: script to run Inception model

##food-fables and food-fables-backend
files for UI 
