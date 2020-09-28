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
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import itertools
import seaborn as sns

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
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # prep for linear layer
        # flattening)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        x = self.fc3(x)

        return x

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
    # sort based on area so we can delete the largest contour which will be the border
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x))

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

#normalise by max rgb
words = np.array(words,dtype="float32")/255.0
print(words.shape)
print(len(words))
words = words.reshape(len(words),1,width,height)
print(words.shape)
print(len(writing_type))
# print(labels)

#gets rid of warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

words_t = torch.Tensor(words)
#another normalisation step. cant think of the actual name (if there is one)
words_t = words_t - torch.mean(words_t, dim=0)
words_t = words_t / torch.std(words_t, dim=0)
le = preprocessing.LabelEncoder()
targets = le.fit_transform(writing_type)
targets = torch.Tensor(targets)

word_dataset = data.TensorDataset(words_t,targets)
word_loader = data.DataLoader(word_dataset, batch_size = 64, num_workers = 0, shuffle = True)


#reproducibility
torch.manual_seed(0)
# Hyperparameters
num_epochs = 200
learning_rate = 0.01

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
print(len(val_loader))
print(len(val_loader))


for epoch in range(num_epochs):
    running_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        # Run the forward pass
        outputs = model(images)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform SGD optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        train_acc += correct/total
        running_loss += loss.item()

    else:
        val_loss = 0
        accuracy = 0

        with torch.no_grad():
            model.eval()
            for images, labels in val_loader:
                outputs = model(images)
                labels = labels.long()
                # Track the accuracy
                total = labels.size(0)
                val_loss += criterion(outputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy += correct/total
                acc_list.append(correct / total)
        model.train()

        train_losses.append(running_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        train_accuracy.append(train_acc/len(train_loader))
        val_accuracy.append(accuracy/len(val_loader))

        print("Epoch: {}/{}.. ".format(epoch+1, num_epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Val Loss: {:.3f}.. ".format(val_losses[-1]),
              "Val Accuracy: {:.3f}".format(accuracy/len(val_loader)))

x = np.arange(num_epochs) + 1
plt.figure()
plt.plot(x,train_losses, 'r-', label = "Train")
plt.plot(x,val_losses, 'b-', label = "Validation")
plt.title("Learning Curve")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
probabilities = 0

with torch.no_grad():
    model.eval()
    test_loss = 0
    accuracy = 0
    i = 0
    for images, labels in test_loader:
        outputs = model(images)
        labels = labels.long()
        # Track the accuracy
        total = labels.size(0)
        test_loss += criterion(outputs, labels)
        probs, predicted = torch.max(outputs.data, 1)
        probs = torch.sigmoid(probs)
        correct = (predicted == labels).sum().item()
        accuracy += correct / total
        acc_list.append(correct / total)
        test_losses.append(test_loss / len(test_loader))
        if i == 0:
            l = labels
            p = predicted
            probabilities = probs
        else:
            l = torch.cat((l,labels),0)
            p = torch.cat((p,predicted),0)
            probabilities = torch.cat((probabilities,probs),0)
        i = i + 1
    x = np.arange(len(probabilities)) + 1
    cm = confusion_matrix(l,p, labels = [0,1,2,3,4,5,6,7,8,9])
    print(p)
    print(l)
    print(cm)
    plot_confusion_matrix(cm,classes = [1,2,3,4,5,6,7,8,9,10])
    print("Test Loss: {:.3f}.. ".format(test_losses[-1]),
        "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))
