#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system(' pip install tensorflow==2.5.0')


# In[2]:


import os
from keras import layers
import keras
import keras.backend as K
import numpy as np
import cv2
from keras.optimizers import Adam, RMSprop


# In[3]:


width = 64
height = 64
channels = 1


# In[4]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

# Encoder
x = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
x = layers.LeakyReLU(name='leaky_1')(x)

x = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(name='leaky_2')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_3', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(name='leaky_3')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_4', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(name='leaky_4')(x)

x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)

g_e = keras.models.Model(inputs=input_layer, outputs=x)

g_e.summary()


# In[5]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

x = g_e(input_layer)

y = layers.Dense(width * width * 2, name='dense')(x) # 2 = 128 / 8 / 8
y = layers.Reshape((width//8, width//8, 128), name='de_reshape')(y)

y = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', name='deconv_1', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_1')(y)

y = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='deconv_2', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_2')(y)

y = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='deconv_3', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_3')(y)

y = layers.Conv2DTranspose(channels, (1, 1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer = 'l2', activation='tanh')(y)

g = keras.models.Model(inputs=input_layer, outputs=y)

g.summary()


# In[6]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

z = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='encoder_conv_1', kernel_regularizer = 'l2')(input_layer)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='encoder_conv_2', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_1')(z)
z = layers.LeakyReLU()(z)


z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='encoder_conv_3', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_2')(z)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_41', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_3')(z)
z = layers.LeakyReLU()(z)

z = layers.GlobalAveragePooling2D(name='encoder_output')(z)

encoder = keras.models.Model(input_layer, z)
encoder.summary()


# In[7]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

f = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='f_conv_1', kernel_regularizer = 'l2')(input_layer)
f = layers.LeakyReLU(name='f_leaky_1')(f)

f = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='f_conv_2', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_1')(f)
f = layers.LeakyReLU(name='f_leaky_2')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_3', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_2')(f)
f = layers.LeakyReLU(name='f_leaky_3')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_4', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_3')(f)
f = layers.LeakyReLU(name='feature_output')(f)

feature_extractor = keras.models.Model(input_layer, f)

feature_extractor.summary()


# In[8]:


class AdvLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class CntLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - encoder(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
# model for training
input_layer = layers.Input(name='input', shape=(height, width, channels))
gan = g(input_layer) # g(x)

adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss])

# loss function
def loss(yt, yp):
    return yp

losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
}

lossWeights = {'cnt_loss': 20.0, 'adv_loss': 1.0, 'enc_loss': 1.0}

# compile
op= Adam(lr=0.0002)
gan_trainer.compile(optimizer = op, loss=losses, loss_weights=lossWeights)


# In[9]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

f = feature_extractor(input_layer)

d = layers.GlobalAveragePooling2D(name='glb_avg')(f)
d = layers.Dense(1, activation='sigmoid', name='d_out')(d)
    
d = keras.models.Model(input_layer, d)
d.summary()


# In[10]:


op= Adam(lr=0.00008)
d.compile(optimizer=op, loss='binary_crossentropy')


# In[11]:


IMAGE_SIZE=64
D = "D:\\harden\\anoGan2\\AnoGAN-MvTec-grid--main\\data01\\bottle\\train\\good\\"
E = "D:\\harden\\anoGan2\\AnoGAN-MvTec-grid--main\\data01\\bottle\\test\\broken_large\\"
#test_final = "E:\\workspace\\opencv_class\\final_test\\test\\"



images = []
fail=[]
labels = []
fail_label=[]
dir_counts = 0
def d1 (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        #print("I:",D+i)
        #cv2.imread(D+i,0)
        try:
            #print("P")
            img1 = cv2.imread(D+i,0)
            #print("P")
            img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #print("P")
            #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts)
        except:
            print("error")
        vou +=1
        if vou >=2000:
            break
    print("A already read")
    return(images,labels)

def d2 (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        #print("I:",D+i)
        #cv2.imread(D+i,0)
        try:
            #print("P")
            img1 = cv2.imread(D+i,0)
            #print("P")
            img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #print("P")
            #img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts+1)
        except:
            print("error")
        vou +=1
        if vou >=2000:
            break
    print("A already read")
    return(images,labels)


# In[12]:


test_1,test_2=d2(E,images=fail,labels=fail_label)
#test_1,test_2=d1(D,images=fail,labels=fail_label)
train_x1,trainx2=d1(D)
train_x1,trainx2=np.expand_dims(train_x1,axis=-1),np.expand_dims(trainx2,axis=-1) 


# In[13]:


len(train_x1)


# In[14]:


test_1,test_2=np.expand_dims(test_1,axis=-1),np.expand_dims(test_2,axis=-1)


# In[15]:


len(test_1)


# In[16]:


from sklearn.model_selection import train_test_split
label = np.array(labels)
X_train_img,X_test_img,y_train_label,y_test_label =  train_test_split(train_x1, trainx2,test_size=0.1,random_state=42 )#
X_train = np.array(X_train_img, dtype=np.float32)
X_test = np.array(X_test_img, dtype=np.float32)
#print("X_train.shape",X_train.shape)
x_train_std = X_train  /255.0#/255.0     / 127 - 1
x_test_std  =  X_test  /255.0#/255.0 / 127 - 1
x_ok = x_train_std
x_test = x_test_std 
#y_trainOneHot = np_utils.to_categorical(y_train_label)
#y_testOneHot = np_utils.to_categorical(y_test_label)
print("x_train_std.shape",x_train_std.shape)
#print("y_train_label",y_train_label.shape)


# In[17]:


from keras.datasets import mnist
import cv2
import numpy as np

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_ok = x_train[y_train == 1] # 6742 筆
# x_test = x_test[(y_test == 7) | (y_test == 1)] # 1135 筆 "1", 1028 筆 "7"
# y_test = y_test[(y_test == 7) | (y_test == 1)]

# def reshape_x(x):
#     new_x = np.empty((len(x), width, height))
#     for i, e in enumerate(x):
#         new_x[i] = cv2.resize(e, (width, height))
#     return np.expand_dims(new_x, axis=-1) / 127 - 1
  
# x_ok = reshape_x(x_ok)
# x_test = reshape_x(x_test)


# In[18]:


x_ok.max()


# In[19]:


niter = 20000
bz = 64


# In[20]:


def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y]


# In[21]:


train_data_generator = get_data_generator(x_ok, bz)


# In[ ]:





# In[ ]:


for i in range(niter):
    
    ### get batch x, y ###
    x, y = train_data_generator.__next__()
        
    ### train disciminator ###
    d.trainable = True
        
    fake_x = g.predict(x)
        
    d_x = np.concatenate([x, fake_x], axis=0)
    d_y = np.concatenate([np.zeros(len(x)), np.ones(len(fake_x))], axis=0)
        
    d_loss = d.train_on_batch(d_x, d_y)

    ### train generator ###
    
    d.trainable = False        
    g_loss = gan_trainer.train_on_batch(x, y)
    
    if i % 500 == 0:
        print(f'niter: {i+1}, g_loss: {g_loss}, d_loss: {d_loss}')


# In[ ]:


'''
# encoded = g_e.predict(x_test)
encoded = g_e.predict(test_1)
# gan_x = g.predict(x_test)
gan_x = g.predict(test_1)
encoded_gan = g_e.predict(gan_x)
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
'''


# In[ ]:


encoded = g_e.predict(x_test)
encoded_gan = g_e.predict(g.predict(x_test))
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~


# In[ ]:


print(score)


# In[ ]:


back=[]
wwww=[]
for i in score:
    if i <0.5:
        back.append("B")
    else:
        wwww.append("W")


# In[ ]:


from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 5
plt.scatter(range(len(test_1)), score, c=['skyblue' if x == 1 else 'pink' for x in test_2]) #y_test])


# In[ ]:


def plot_image(image,labels,prediction,idx,num=10):  

    fig = plt.gcf() 

    fig.set_size_inches(18, 24) 

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


# In[ ]:


re = d.predict(test_1)#gan_x test_1[:10]
#print("re",re==1)
back=[]
wwww=[]
for i in re:
    if i == 1:
        back.append("B")
    else:
        wwww.append("W")
#print("la",test_2[2000:2010])


# In[ ]:


print(len(test_1))
len(back)


# In[ ]:


gan_x=gan_x*255
plot_image(gan_x,test_2,score,10)


# In[ ]:


test_1=test_1*255
plot_image(test_1,test_2,score,10)


# In[ ]:


i = 1 # or 1
image = np.reshape(gan_x[i:i+1], (64, 64))
image = image *255 #+ 127
plt.imshow(image.astype(np.uint8), cmap='gray')


# In[ ]:


image = np.reshape(test_1[i:i+1], (64, 64))#x_test
image = image * 255 #+ 127
plt.imshow(image.astype(np.uint8), cmap='gray')


# In[ ]:


ori=np.reshape(test_1[i:i+1], (64, 64,1))
sim=np.reshape(gan_x[i:i+1], (64, 64,1))
np_residual =ori-sim

np_residual = (np_residual + 2)/4
np_residual=np_residual.astype(np.uint8)
residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)
print(residual_color.shape)
print(ori.shape)
show = cv2.addWeighted(ori, 0.3,residual_color, 0.7, 0.)
plt.imshow(show.astype(np.uint8), cmap='gray')


# In[ ]:





# In[ ]:




