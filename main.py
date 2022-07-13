from __future__ import print_function

import matplotlib
matplotlib.use('Qt5Agg')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import argparse
import anogan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=2)
parser.add_argument('--label_idx', type=int, default=6)
parser.add_argument('--mode', type=str, default='test', help='train, test')
args = parser.parse_args()
'''
### 0. prepare data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]

X_test_original = X_test.copy()

X_train = X_train[y_train==1]
X_test = X_test[y_test==1]
print ('train shape:', X_train.shape)

'''
### data 
D = "D:\\harden\\02\\simpsons_dataset\\bart_simpson\\"
images = []
labels = []
dir_counts = 0
IMAGE_SIZE = 112
def d (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        
        try:
            img1 = cv2.imread(D+i)
            img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts)
        except:
            print("error")
        vou +=1
        if vou >=500:
            break
    print("A already read")
    return(images,labels)
d(D,images,labels)
label = np.array(labels)
images = np.array(images)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(images, label,test_size=0.1,random_state=42 )#

X_train = (X_train.astype(np.float32) /255.0) #- 127.5) / 127.5
X_test = (X_test.astype(np.float32) /255.0) #- 127.5) / 127.5

#X_train = X_train[:,:,:,None]
#X_test = X_test[:,:,:,None]

X_test_original = X_test.copy()

print ('train shape:', X_train.shape)



### 1. train generator & discriminator
if args.mode == 'train':
    Model_d, Model_g = anogan.train(3, X_train)

### 2. test generator
generated_img = anogan.generate(5)
img = anogan.combine_images(generated_img)
img = (img*255.0)#127.5)+127.5
img = img.astype(np.uint8)
img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

### opencv view
# cv2.namedWindow('generated', 0)
# cv2.resizeWindow('generated', 256, 256)
# cv2.imshow('generated', img)
# cv2.imwrite('result_latent_10/generator.png', img)
# cv2.waitKey()

### plt view
# plt.figure(num=0, figsize=(4, 4))
# plt.title('trained generator')
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# exit()

### 3. other class anomaly detection

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 112, 112, 3), iterations=500, d=d)

    # anomaly area, 255 normalization
    np_residual = test_img.reshape(112,112,3) - similar_img.reshape(112,112,3)
    np_residual = (np_residual + 2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(112,112,3)*255.0).astype(np.uint8) #*127.5+127.5
    similar_x = (similar_img.reshape(112,112,3)*255.0).astype(np.uint8) #*127.5+127.5

    original_x_color = original_x # cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

    return ano_score, original_x, similar_x, show, residual_color


### compute anomaly score - sample from test set
# test_img = X_test_original[y_test==1][30]

### compute anomaly score - sample from strange image
# test_img = X_test_original[y_test==0][30]

### compute anomaly score - sample from strange image
img_idx = args.img_idx
label_idx = args.label_idx
# test_img = X_test_original[y_test==label_idx][img_idx]
test_img = X_test_original[img_idx]
# test_img = np.random.uniform(-1,1, (112,112,1))

start = cv2.getTickCount()
score, qurey, pred, diff,residual_color = anomaly_detection(test_img)
time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
# cv2.imwrite('./qurey.png', qurey)
# cv2.imwrite('./pred.png', pred)
# cv2.imwrite('./diff.png', diff)

## matplot view
plt.figure(1, figsize=(3, 3))
plt.title('query image')
plt.imshow(qurey.reshape(112,112,3))#, cmap=plt.cm.gray)

print("anomaly score : ", score)
plt.figure(2, figsize=(3, 3))
plt.title('generated similar image')
plt.imshow(pred.reshape(112,112,3))#, cmap=plt.cm.gray)

plt.figure(3, figsize=(3, 3))
plt.title('anomaly detection')
plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
plt.show()

plt.figure(4, figsize=(3, 3))
plt.title('residual')
plt.imshow(residual_color)
plt.show()
### 4. tsne feature view

### t-SNE embedding 
### generating anomaly image for test (radom noise image)

from sklearn.manifold import TSNE

random_image = np.random.uniform(0, 1, (100, 112, 112, 3))
print("random noise image")
plt.figure(4, figsize=(2, 2))
plt.title('random noise image')
plt.imshow(random_image[0].reshape(112,112,3), cmap=plt.cm.gray)

# intermidieate output of discriminator
model = anogan.feature_extractor()
feature_map_of_random = model.predict(random_image, verbose=1)
feature_map_of_minist = model.predict(X_test_original[y_test != 1][:300], verbose=1)
feature_map_of_minist_1 = model.predict(X_test[:100], verbose=1)

# t-SNE for visulization
output = np.concatenate((feature_map_of_random, feature_map_of_minist, feature_map_of_minist_1))
output = output.reshape(output.shape[0], -1)
anomaly_flag = np.array([1]*100+ [0]*300)

X_embedded = TSNE(n_components=2).fit_transform(output)
plt.figure(5)
plt.title("t-SNE embedding on the feature representation")
plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
plt.scatter(X_embedded[100:400,0], X_embedded[100:400,1], label='mnist(anomaly)')
plt.scatter(X_embedded[400:,0], X_embedded[400:,1], label='mnist(normal)')
plt.legend()
plt.show()