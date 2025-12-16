#!/usr/bin/env python
# coding: utf-8

# # screen capture

# In[7]:


def one_time():
    import pyscreenshot as ImageGrab
    import time
    images_folder = "C:/Users/mmmki/OneDrive/Documents/cv_project/captured_images/9/"
    for i in range(0,101):
        time.sleep(8)
        im=ImageGrab.grab(bbox=(60,170,400,550))
        print("saved......",i)
        im.save(images_folder+str(i)+'.png')
        print("clear screen now and redraw now......")


# In[10]:





# # Generate Dataset

# In[12]:


import cv2
import csv
import glob

header =["label"]
for i in range(0,64):
    header.append("pixel"+str(i))
with open('dataset.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for label in range(10):
    dirList = glob.glob("C:/Users/mmmki/OneDrive/Documents/cv_project/captured_images/"+str(label)+"/*.png")

    for img_path in dirList:
        im= cv2.imread(img_path)
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray,(5,5),0)
    roi= cv2.resize(im_gray,(8,8), interpolation=cv2.INTER_AREA)

    data=[]
    data.append(label)
    rows, cols = roi.shape

    ##add pixel one by one into data array
    for i in range(rows):
        for j in range(cols):
            k=roi[i,j]
            if k>100:
                k=1
            else:
                k=0
            data.append(k)
    with open('dataset.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(data)


# # Load the dataset

# In[13]:


import pandas as pd
from sklearn.utils import shuffle
data = pd.read_csv('dataset.csv')
data = shuffle(data)
data


# # separation of dependent and independent variable

# In[14]:


x = data.drop(["label"],axis=1)
y = data["label"]


# # preview of one image using matplotlib

# In[26]:


import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# تحميل الموديل
model = joblib.load("C:/Users/mmmki/OneDrive/Documents/cv_project/model/digit_recognizer")

# افترض آخر صورة من dataset
idx = len(x_clean) - 1
img = x_clean.iloc[idx].values.astype(np.uint8).reshape(8, 8)

# --- Preprocessing زي ما اتدرب الموديل ---
# إذا الرقم أسود على أبيض في الصور الحالية: نعكس اللون
roi = cv2.bitwise_not(img)

# scale القيم لتكون بين 0-16 زي ما اتدرب
roi = roi.astype(np.float32) / 16.0

# flatten وتحويله لـ DataFrame بنفس الأعمدة زي training
x_input = roi.flatten().reshape(1, -1)
x_input_df = pd.DataFrame(x_input, columns=x_clean.columns)

# --- Prediction ---
prediction = model.predict(x_input_df)

# --- عرض الصورة مع النتائج ---
plt.figure(figsize=(3,3))
plt.imshow(img, cmap='gray')
plt.title(f"True: {y_clean.iloc[idx]}, Predicted: {prediction[0]}")
plt.axis('off')
plt.show()

print("True label:", y_clean.iloc[idx])
print("Prediction:", prediction[0])


# # Train-test split

# In[8]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, test_size = 0.2)


# In[22]:


import joblib
from sklearn.svm import SVC 



from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()
X = digits.data       
y = digits.target     


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=6)



classifier=SVC(kernel="linear", random_state=6) 
classifier.fit(train_x,train_y) 
joblib.dump(classifier,"C:/Users/mmmki/OneDrive/Documents/cv_project/model/digit_recognizer")


# # calculate accuracy

# In[23]:


from sklearn import metrics
prediction=classifier.predict(test_x)
print("Accuracy= ",metrics.accuracy_score(prediction, test_y))


# # prediction of image drawn in paint
# 

# In[3]:


import joblib 
import cv2 
import numpy as np 
import time 
import pyscreenshot as ImageGrab 
import os 
from PIL import ImageGrab
images_folder = "img/" 
if not os.path.exists(images_folder): os.makedirs(images_folder) 
img = ImageGrab.grab(bbox=(60,170,400,500)) 
img.save(images_folder + "img.png") 
model=joblib.load("C:/Users/mmmki/OneDrive/Documents/cv_project/model/digit_recognizer") 
image_folder="img/" 
while True: 
    img=ImageGrab.grab(bbox=(60,170,400,500)) 
    img.save(images_folder+"img.png") 
    im = cv2.imread(images_folder+"img.png") 
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
    im_gray = cv2.GaussianBlur(im_gray, (5,5), 0) 
    #threshold the image 
    ret, im_th = cv2.threshold(im_gray,100,255, cv2.THRESH_BINARY) 
    roi= cv2.resize(im_th, (8,8), interpolation =cv2.INTER_AREA) 
    rows,cols=roi.shape 

    ##add pixel one by one into data array 
    x = [] 
    for i in range(rows): 
        for j in range(cols): 
            k = roi[i,j] 
            if k>100: 
                k=1 
            else:
                k=0 
            x.append(k) 
    roi = roi / 16.0
    x = roi.flatten().reshape(1, -1)        
    prediction =model.predict(x)
    print("Prediction: ",prediction) 
    cv2.putText(im, "prediction is: "+str(prediction[0]),(20,20),0,0.8,(0,255,0),2,cv2.LINE_AA) 
    cv2.startWindowThread() 
    cv2.namedWindow("Result") 
    cv2.imshow("Result",im) 
    cv2.waitKey(10000) 
    if cv2.waitKey(1)==13: 
        break 
cv2.destroyAllWindows()


# In[ ]:







