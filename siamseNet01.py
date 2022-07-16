from __future__ import absolute_import
from __future__ import print_function
import numpy as np
 
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop,Adam
from keras import backend as K
 
num_classes = 2
epochs = 3000
 
 
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
 
 
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
 
 
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
 
 
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
 
 
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)
 
 
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
 
 
def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
 
'''
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
'''
IMAGE_SIZE = 64
import cv2
import os
D = "C:\\Users\\User\\Desktop\\white_smoke\\ss\\"
E = "E:\\workspace\\project_\\vv\\"
images1 = []
labels1 = []
dir_counts = 0
def d (D=D,images=images1,labels=labels1):
    vou=0
    for i in os.listdir(D):
        
        try:
            #print(D+i)
            img1 = cv2.imread(D+i)
            #print(D+i)
            img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            #print(0)
            labels.append(dir_counts)
        except:
            print("error")
        vou +=1
        if vou >=2000:
            break
    print("A already read")
    return(images,labels)
d(E,images1,labels1)
test_00b = images1[0]
def e (E=E,images=images1,labels=labels1):
    BC = 0
    for i in os.listdir(E):
        try:
            img2 = cv2.imread(E+i)
            img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
            #img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img2)
            labels.append(dir_counts+1)
        except:
            print("error")
        BC = BC+1
        if BC == 2000:
            break
    print("B already read")
    return(images,labels)
e(D,images1,labels1)
print("LAB",labels1)
from sklearn.model_selection import train_test_split
label = np.array(labels1)
X_train,X_test,y_train,y_test =  train_test_split(images1, label,test_size=0.1,random_state=42 )#
x_train = np.array(X_train, dtype=np.float32)
x_test = np.array(X_test, dtype=np.float32)
x_train = x_train/255.0
x_test  =  x_test/255.0
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
tr_y=tr_y.astype(float)
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
print("digit_indices",digit_indices)
te_pairs, te_y = create_pairs(x_test, digit_indices)
te_y=te_y.astype(float)
# network definition
input_shape=(64,64,3)
base_network = create_base_network(input_shape)
 
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

print("len len(te_pairs)",len(te_pairs))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)
 
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
 
model = Model([input_a, input_b], distance)
 
# train
rms = RMSprop()
adam =Adam(lr=0.00009)
model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=32,#128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
 
# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
 
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

print("shape te_pairs",te_pairs.shape)

import cv2
from keras.models import load_model
# predict data with model
def predict(model_path, image_path1, image_path2, target_size,i):
    #saved_model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss})
    saved_model = model_path
    # image1 = cv2.imread(image_path1)
    # image2 = cv2.imread(image_path2)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.resize(image1, target_size)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.resize(image2, target_size)  # <class 'numpy.ndarray'>
    image1 = image_path1
    image2 = image_path2
    # print(image2.shape)  # (80, 80)
    # print(image2)
    data1 = np.array([image1], dtype='float') #/ 255.0 / 255.0
    data2 = np.array([image2], dtype='float') #/ 255.0 / 255.0
    #print(data1.shape, data2.shape)  # (1, 80, 80) (1, 80, 80)
    pairs = np.array([data1, data2])
    #print(pairs.shape)  # (2, 80, 80)
    
    y_pred = saved_model.predict([data1, data2])
    print(y_pred)
    # print(y_pred)  # [[4.1023154]]
    pred = y_pred.ravel() < 0.5
    print("第 %d 個 pred: " % i ,pred)  # 相似程度
    y_true = [1]  # 1表示同一類, 0表示不同類
    if pred == y_true:
        print("視同一類")
    else:
        print("不是同一類")
    return pred

pre=[]
for i in range(len(te_pairs)):
    score=predict(model,te_pairs[i,1],te_pairs[i, 0],target_size=(64,64),i=i)
    pre.append(score)

import matplotlib.pyplot as plt


#img1=(test_00b ).astype(int)
#img2=(te_pairs[-3, 0] *255).astype(int)
#print(te_pairs.shape)

#plt.imshow(img1)
#plt.show()

#plt.imshow(img2)
#plt.show()
#####
def plot_image(image,labels,prediction,idx,num=20):  
    fig = plt.gcf() 
    
    fig.set_size_inches(12, 14) 
    if num>25: 
        num=25 
    for i in range(0, num): 
        ax = plt.subplot(5,5, 1+i) 
        ax.imshow(image[idx], cmap='binary') 
        title = "label=" +str(labels[idx]) 
        if len(prediction)>0: 
            title+=",perdict="+str(prediction[idx]) 
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([]) 
        idx+=1 
    plt.show() 
plot_image(te_pairs[:,1],te_y[:],pre[:],idx=0,num=len(te_pairs))

plot_image(te_pairs[:,0],te_y[:],pre[:],idx=0,num=len(te_pairs))

