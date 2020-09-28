# Honours-Thesis
Scripts used for my honours analysis, however the code is not modularised and is saved/ran entirely through one script. I know this is bad practice, I was just lazy.

Cnn.py is a python script that includes my image processing script, using openCv; a definition of my CNN model that is made using PyTorch and the training algorithm; and my code that was used to run inference on the testing dataset and plotting the learning/accuracy curves and confusion matrix. Similarly, siamese.py is set up in a similar fashion to Cnn.py, however the definition of our model is changed to run two CNN's in parallel. 
