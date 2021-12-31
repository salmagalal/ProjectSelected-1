# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 21:21:28 2021

@author: DELL
"""

import os 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = 'G:\\FCAI\\Level 3\\Semester1\\Selected-1\\ProjectSVM\\data\\natural_images'
categories = ['airplane' , 'car' , 'cat' , 'dog' , 'flower' , 'fruit' , 'motorbike' , 'person']
data = []


"""for category in categories:
    path = os.path.join(dir,category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        natural_img = cv2.imread(imgpath,0)
        cv2.imshow('image',natural_img)
        break
    break
cv2.waitKey(0)
cv2.destroyAllWindows()"""


for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        natural_img = cv2.imread(imgpath,0)
        try:
            natural_img = cv2.resize(natural_img,(50,50))
            image = np.array(natural_img).flatten()
        
            data.append([image,label])
        except Exception as e: 
            pass
        
print(len(data))



pick_in = open('data1.pickle' , 'wb')
pickle.dump(data,pick_in)
pick_in.close()

pick_in = open('data1.pickle' , 'rb')
data = pickle.load(pick_in)
pick_in.close()


random.shuffle(data)
features= []
labels = []

for feature ,label in data:
    features.append(feature)
    labels.append(label)
#print(len(labels))
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)

model = SVC(C=1,kernel='poly',gamma='auto', probability=(True))
model.fit(xtrain, ytrain)

pick = open('model.sav','wb')
pickle.dump(model,pick)
pick.close()


pick = open('model.sav','rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, prediction)
print(cm)

categories = ['airplane' , 'car' , 'cat' , 'dog' , 'flower' , 'fruit' , 'motorbike' , 'person']

print('Accuracy: ', accuracy)
print('Prediction is : ',categories[prediction[0]])
 
myimg = xtest[0].reshape(50,50)
plt.imshow(myimg,cmap='gray')
plt.show()

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(xtest)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(ytest, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# method II: ggplot
from ggplot import *
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')
