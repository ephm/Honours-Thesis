import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import itertools
import random
import time
import seaborn as sns
import pandas as pd

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # input size 224 x 224
        # conv maps to 220
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)

        # maxpool layer with F = 2, S = 2
        # max pool converts to (32, 110, 110)
        self.pool = nn.MaxPool2d(2,2)

        # second conv layer: 32 inputs, 64 outputs , 3x3 conv
        # conv converts to 108
        # max pool converts to (64, 54, 54)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)

        # third conv layer: 64 inputs, 128 outputs , 3x3 conv
        # conv converts to 52
        # output dimension: (128, 52, 52)
        # maxpool converts to (128, 26, 26)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)

        self.fc_drop = nn.Dropout(p = 0.5)

        #global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d((20,1))
        # 64 outputs * 5x5 filtered/pooled map  = 186624
        self.fc1 = nn.Linear(128*20, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 11)

    #create an instance of a single CNN
    def forward_once(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # prep for linear layer
        # flattening
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)
        return x

    #run two models in parallel
    def forward(self, in1, in2):
        out1 = self.forward_once(in1)
        out2 = self.forward_once(in2)
        return out1, out2

#working as intended
def convertToWord(original_image):
    image = original_image.copy()

    # load the image, convert it to grayscale, and blur it to remove noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((5, 20), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=3)

    # finding contours
    contours, hierachy = cv2.findContours(img_dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_of_interest = []
    print(len(contours))

    for cnt in contours:
        # hopefully gets rid of dotted i's
        if cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 1000000:
            # get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            roi = image[y:y + h, x:x + w]
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (90, 0, 255), 2)
            regions_of_interest.append(roi)
    return regions_of_interest


def siameseData(x,y,n):
    # takes in a dataframe, and length of siamese database you want
    # returns a siamese database i.e. tuples (img1, img2, label) where
    # label can either be 0 for dif hadnwriters or 1 for same
    label = random.randint(0, 1)

    # CAN ADD A WHILE LOOP THAT CHECKS IF THE INDX's ARE THE SAME
    # IN CASE WE CANT HAVE SAME HANDWRITEN WORD NEXT TO EACH OTHER
    randLabel = random.randint(1,n)
    label1 = randLabel
    indices = [i for i, labels in enumerate(y) if labels == randLabel]
    indx = random.choice(indices)
    img1 = x[indx]
    if label == 1:
        # create pairs of same handwriter
        # locate 2 images with the same lab from df
        # sample a row from all the labels
        indx2 = random.choice(indices)
        label2 = label1
        img2 = x[indx2]
    else:
        # pairs of different handwriters
        indices1 = [i for i, labels in enumerate(y) if labels != randLabel]
        indx3 = random.choice(indices1)
        label2 = y[indx3]
        img2 = x[indx3]

    return img1, img2, label1, label2

writing_type = []
words = []
#load in images and process them into a sorted vector so we will have writer 1,2,3... etc
filenames = glob.glob("*.tif")
filenames.sort()
images = [cv2.imread(img,1) for img in filenames]
i = 0
count = 0
#create a database of words with their respective writer labels
for img in images:
    if i%4==0:
        count = count + 1
    label = count
    temp = convertToWord(img)
    words += temp
    for k in temp:
        writing_type.append(label)
    i = i+1

width = 224
height = 224
for i in range(len(words)):
     words[i] = cv2.resize(words[i], (width,height), interpolation = cv2.INTER_CUBIC)

#normalise and reshape images to be 4 dimensions for pytorch
words = np.array(words,dtype="float32")/255.0
words = words.reshape(len(words),1,width,height)
random.seed(0)

#Create siamese word pairs
pair_match = []
pair_match2 = []
img1, img2, lab1, lab2 = siameseData(words,writing_type,10)
pair_match.append(lab1)
pair_match2.append(lab2)
pair1 = np.asarray(img1,dtype="float32")
pair2 = np.asarray(img2,dtype="float32")
TicToc = TicTocGenerator() # create an instance of the TicTocGen generator
tic()

for i in range(1,4000):
    img1, img2, lab1, lab2 = siameseData(words,writing_type,10)
    img1 = np.asarray(img1, dtype="float32")
    img1 = img1 / 255.0
    img1 = img1 - np.mean(img1, axis = (0,1,2), keepdims = True)
    img1 = img1 / np.std(img1, axis = (0,1,2), keepdims = True)

    img2 = np.asarray(img2, dtype="float32")
    img2 = img2 / 255.0
    img2 = img2 - np.mean(img2, axis = (0,1,2), keepdims = True)
    img2 = img2 / np.std(img2, axis = (0,1,2), keepdims = True)

    pair1 = np.vstack((pair1,img1))
    pair2 = np.vstack((pair2,img2))
    pair_match.append(lab1)
    pair_match2.append(lab2)


#converts pair1 and pair2 into a 4 dimension vector (batch size, channels, width,height): channels = 1 coz grayscale
pair1 = pair1.reshape(len(pair1),1,width,height)
pair2 = pair2.reshape(len(pair2),1,width,height)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

words1 = torch.Tensor(pair1)
#another normalisation step. cant think of the actual name (if there is one)


words2 = torch.Tensor(pair2)
#another normalisation step. cant think of the actual name (if there is one)

le = preprocessing.LabelEncoder()
targets = le.fit_transform(pair_match)
targets = torch.Tensor(pair_match)
targets1 = le.fit_transform(pair_match2)
targets1 = torch.Tensor(pair_match2)

word_dataset = data.TensorDataset(words1,words2,targets,targets1)
word_loader = data.DataLoader(word_dataset, batch_size = 64, num_workers = 0, shuffle = True)

#reproducibility
torch.manual_seed(0)
# Hyperparameters
num_epochs = 250
num_classes = 2
batch_size = 100
learning_rate = 0.01

#create a 60/20/20 train/val/test split
train_size = int(0.6 * len(word_dataset))
val_size = len(word_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(word_dataset, [train_size, val_size])
print(len(train_dataset))
print(len(val_dataset))
val_size = int(val_size*0.5)
test_size = len(val_dataset) - val_size
print(val_size)
print(test_size)
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [test_size, val_size])

#convert data to loaders to run pytorch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)


model = ConvNet()
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
train_losses, val_losses, test_losses = [], [], []
train_accuracy, val_accuracy = [], []

print(len(train_loader))

print(len(test_loader))

for epoch in range(num_epochs):
    train_acc = 0
    running_loss = 0
    correct = 0
    for images1, images2, labels, labels1 in train_loader:
        # Run the forward pass
        output1, output2 = model(images1, images2)
        labels = labels.long()
        labels1 = labels1.long()
        loss1 = criterion(output1,labels)
        loss2 = criterion(output2,labels1)
        # loss = criterion(output1, output2, labels)
        total = labels.size(0)

        _, pred1 = torch.max(output1, 1)
        _, pred2 = torch.max(output2, 1)
        predicted = (pred1 == pred2)
        lab = (labels == labels1)
        correct = (predicted == lab).sum().item()

        train_acc += correct / total
        loss = (loss1 + loss2)/2
        loss_list.append(loss.item())

        # Backprop and perform SGD optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        val_loss = 0
        accuracy = 0
        correct = 0

        with torch.no_grad():
            model.eval()
            for images1, images2, labels, labels1 in val_loader:
                output1, output2 = model(images1, images2)
                # outputs = torch.sigmoid(outputs)
                labels = labels.long()
                labels1 = labels1.long()

                # Track the accuracy
                total = labels.size(0)
                loss1 = criterion(output1, labels)
                loss2 = criterion(output2, labels1)
                val_loss += (loss1+loss2)/2
                # test_loss += criterion(output1, output2, labels)

                _, pred1 = torch.max(output1, 1)
                _, pred2 = torch.max(output2, 1)
                predicted = (pred1 == pred2)
                lab = (labels == labels1)
                correct = (predicted == lab).sum().item()
                # train_accuracy += correct / total
                accuracy += correct/total
                acc_list.append(correct / total)
        model.train()

        train_accuracy.append(train_acc/len(train_loader))
        val_accuracy.append(accuracy/len(val_loader))
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))

        print("Epoch: {}/{}.. ".format(epoch+1, num_epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Train Accuracy: {:.3f}..".format(train_accuracy[-1]),
              "Validation Loss: {:.3f}.. ".format(val_losses[-1]),
              "Validation Accuracy: {:.3f}".format(accuracy/len(val_loader)))
toc()
x = np.arange(num_epochs) + 1
plt.figure()
plt.plot(x,train_accuracy, 'r-', label = "Train")
plt.plot(x,val_accuracy, 'b-', label = "Validation")
plt.title("Model Accuracy with 4000 Generated Pairs")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

x = np.arange(num_epochs) + 1
plt.figure()
plt.plot(x,train_losses, 'r-', label = "Train")
plt.plot(x,val_losses, 'b-', label = "Validation")
plt.title("Learning Curve with 4000 Generated Pairs")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()



with torch.no_grad():
    model.eval()
    test_loss = 0
    accuracy = 0
    i = 0
    for images1, images2, labels, labels1 in test_loader:
        output1, output2 = model(images1, images2)
        # outputs = torch.sigmoid(outputs)
        d = F.pairwise_distance(torch.sigmoid(output1), torch.sigmoid(output2))
        labels = labels.long()
        labels1 = labels1.long()

        # Track the accuracy
        total = labels.size(0)
        loss1 = criterion(output1, labels)
        loss2 = criterion(output2, labels1)
        test_loss += (loss1 + loss2) / 2

        _, pred1 = torch.max(output1, 1)
        _, pred2 = torch.max(output2, 1)
        predicted = (pred1 == pred2)
        lab = (labels == labels1)
        correct = (predicted == lab).sum().item()
        # train_accuracy += correct / total
        accuracy += correct / total
        acc_list.append(correct / total)
        test_losses.append(test_loss / len(test_loader))
        if i == 0:
            l = lab
            p = predicted
            dist = d
        else:
            l = torch.cat((l,lab),0)
            p = torch.cat((p,predicted),0)
            dist = torch.cat((dist,d),0)
        i = i + 1
    cm = confusion_matrix(l,p, labels = [0,1])
    # df = pd.DataFrame(columns = ["Similarity","Writing type"])
    # sim_scores = []
    # for i in range(len(dist)):
    #     sim = 1/(1+dist[i])
    #     sim_scores.append(sim)
    # print(dist)
    # print(sim_scores)
    # natural = []
    # disguised = []
    # different = []
    # for i in range(len(sim_scores)):
    #     if l[i]:
    #         #by observation, no natural handwriting had a similarity score lower than 0.55
    #         if sim_scores[i] > 0.55:
    #             df.loc[i] = [float(sim_scores[i]), "Natural"]
    #             natural.append(sim_scores[i])
    #         else:
    #             df.loc[i] = [float(sim_scores[i]), "Disguised"]
    #             disguised.append(sim_scores[i])
    #     else:
    #         df.loc[i] = [float(sim_scores[i]), "Different"]
    #         different.append(sim_scores[i])
    # print(df)
    # sns.boxplot(x = "Writing type", y = "Similarity", data=df)
    # plt.title("Similarity Scores for Predictions in Questioned Writings3.")
    # plt.show()
    # # sns.kdeplot(natural, color = "purple", shade = True)
    # # sns.kdeplot(disguised, color = "orange", shade = True)
    # # sns.kdeplot(different, color="green", shade=True)
    # # plt.title("Similarity Scores of Predictions")
    # # plt.xlabel("Similarity Score")
    # # plt.ylabel("Frequency")
    # # plt.legend(["Natural", "Disguised", "Different"])
    # # plt.show()
    plot_confusion_matrix(cm,classes = [0,1])
    print("Test Loss: {:.3f}.. ".format(test_losses[-1]),
        "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))
    fpr, tpr, threshold = metrics.roc_curve(l, p)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
