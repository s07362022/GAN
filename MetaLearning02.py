import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds
import cv2
import os 


"""
## Define the Hyperparameters
"""

learning_rate = 0.003 # 0.003
meta_step_size = 0.25  # 0.25

inner_batch_size = 25 #25
eval_batch_size = 25 #25

meta_iters = 8000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20
shots = 20 #5  every shots support how many data to training and repersent lear vector
classes = 5



class Dataset:
    def __init__(self, training):
        # Download the tfrecord files containing the omniglot data and convert to a
        # dataset.
        split = "train" if training else "test"
        IMAGE_SIZE = 64
        def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
            top, bottom, left, right = (0, 0, 0, 0)
            
            #get size
            h, w , _= image.shape
            
            #adj(w,h)
            longest_edge = max(h, w)    
            
            #size = n*n 
            if h < longest_edge:
                dh = longest_edge - h
                top = dh // 2
                bottom = dh - top
            elif w < longest_edge:
                dw = longest_edge - w
                left = dw // 2
                right = dw - left
            else:
                pass 

            BLACK = [0, 0, 0]   
            constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
            return cv2.resize(constant, (height, width))
        self.data = {}
        self.images = []
        self.labels = []
        
        
        D = "E:\\data\\bottle\\train\\good\\"
        E = "E:\\data\\bottle\\test\\broken_large\\"
        F = "E:\\data\\bottle\\test\\broken_large\\"
        B = "E:\\workspace\\project_\\bad_file\\"
        S = "E:\\workspace\\project_\\file_smoke"
        C = "E:\\workspace\\segmentation\\unt01_OK\\cat"
        def d (D=D,images=self.images,labels=self.labels,dir_counts = "0",vou=0):
            vou=0
            for i in os.listdir(D):
                
                try:
                    img1 = cv2.imread(D+i)
                    #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))/255.0
                    #img1=np.expand_dims(img1,axis=-1)
                    img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE) /255.0
                    images.append(img1)
                    labels.append(dir_counts)
                    label = dir_counts
                    if label not in self.data:
                        self.data[label] = []
                    self.data[label].append(img1)
                    
                except:
                    print("error")
                vou +=1
                if vou >=80:
                    break
            print("A already read")
            return(images,labels)
        if training ==True:
            d(D)
            d(E,images=self.images,labels=self.labels,dir_counts="1")
            d(B,images=self.images,labels=self.labels,dir_counts="2")
            d(S,images=self.images,labels=self.labels,dir_counts="3")
            d(C,images=self.images,labels=self.labels,dir_counts="4")
        else:
            d(D,vou=40)
            d(E,images=self.images,labels=self.labels,dir_counts="1")
            d(B,images=self.images,labels=self.labels,dir_counts="2",vou=40)
            d(S,images=self.images,labels=self.labels,dir_counts="3",vou=40)
            d(C,images=self.images,labels=self.labels,dir_counts="4",vou=40)
            #pass
        #print("data",self.data)
        #print(self.labels)
        from sklearn.model_selection import train_test_split
        self.images,X_test,self.labels,y_test =  train_test_split(self.images, self.labels,test_size=0.4,random_state=42 )


    def get_mini_dataset(
        self, batch_size, repetitions, shots, num_classes, split=False
    ):
        temp_labels = np.zeros(shape=(num_classes * shots))
        temp_images = np.zeros(shape=(num_classes * shots, 64, 64, 3)) # 28 28 3
        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 64, 64, 3)) # 28 28 1

        # Get a random subset of labels from the entire label set.
        label_subset = random.choices(self.labels, k=num_classes)
        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                test_images[class_idx] = images_to_split[-1]
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = images_to_split[:-1]
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                temp_images[
                    class_idx * shots : (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (temp_images.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        if split:
            return dataset, test_images, test_labels
        return dataset


import urllib3

urllib3.disable_warnings()  # Disable SSL warnings that may happen during download.
train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

"""
## Visualize some examples from the dataset
"""

_, axarr = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

sample_keys =  train_dataset.labels #list(train_dataset.data.keys())
print("sample_keys: ", sample_keys)
test_label = test_dataset.labels
print("test_label:", test_label)
#test_images=test_dataset.images ########################################################

for a in range(5):
    for b in range(5):
        temp_image = train_dataset.data[sample_keys[a]][b]
        #print("temp_image: ", temp_image)
        temp_image = np.stack((temp_image[:, :, 0],) * 3, axis=2)
        temp_image *= 255
        temp_image = np.clip(temp_image, 0, 255).astype("uint8")
        if b == 2:
            axarr[a, b].set_title("Class : " + sample_keys[a])
        axarr[a, b].imshow(temp_image, cmap="gray")
        axarr[a, b].xaxis.set_visible(False)
        axarr[a, b].yaxis.set_visible(False)
plt.show()

"""
## Build the model
"""


def conv_bn(x):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


inputs = layers.Input(shape=(64, 64, 3)) # 28 28 1
x = conv_bn(inputs)
x = conv_bn(x)
x = conv_bn(x)
x = conv_bn(x)
x = layers.Flatten()(x)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile()
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

"""
## Train the model
"""

training = []
testing = []
for meta_iter in range(meta_iters):
    frac_done = meta_iter / meta_iters
    cur_meta_step_size = (1 - frac_done) * meta_step_size
    # Temporarily save the weights from the model.
    old_vars = model.get_weights()
    # Get a sample from the full dataset.
    mini_dataset = train_dataset.get_mini_dataset(
        inner_batch_size, inner_iters, train_shots, classes
    )
    for images, labels in mini_dataset:
        with tf.GradientTape() as tape:
            preds = model(images)
            loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    new_vars = model.get_weights()
    # Perform SGD for the meta step.
    for var in range(len(new_vars)):
        new_vars[var] = old_vars[var] + (
            (new_vars[var] - old_vars[var]) * cur_meta_step_size
        )
    # After the meta-learning step, reload the newly-trained weights into the model.
    model.set_weights(new_vars)
    # Evaluation loop
    if meta_iter % eval_interval == 0:
        accuracies = []
        for dataset in (train_dataset, test_dataset):
            # Sample a mini dataset from the full dataset.
            train_set, test_images, test_labels = dataset.get_mini_dataset(
                eval_batch_size, eval_iters, shots, classes, split=True
            )
            old_vars = model.get_weights()
            # Train on the samples and get the resulting accuracies.
            for images, labels in train_set:
                with tf.GradientTape() as tape:
                    preds = model(images)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            test_preds = model.predict(test_images)
            test_preds = tf.argmax(test_preds).numpy()
            num_correct = (test_preds == test_labels).sum()
            # Reset the weights after getting the evaluation accuracies.
            model.set_weights(old_vars)
            accuracies.append(num_correct / classes)
        training.append(accuracies[0])
        testing.append(accuracies[1])
        if meta_iter % 100 == 0:
            print(
                "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
            )

"""
## Visualize Results
"""

# First, some preprocessing to smooth the training and testing arrays for display.
window_length = 100
train_s = np.r_[
    training[window_length - 1 : 0 : -1], training, training[-1:-window_length:-1]
]
test_s = np.r_[
    testing[window_length - 1 : 0 : -1], testing, testing[-1:-window_length:-1]
]
w = np.hamming(window_length)
train_y = np.convolve(w / w.sum(), train_s, mode="valid")
test_y = np.convolve(w / w.sum(), test_s, mode="valid")

# Display the training accuracies.
x = np.arange(0, len(test_y), 1)
plt.plot(x, test_y, x, train_y)
plt.legend(["test", "train"])
plt.grid()

train_set, test_images, test_labels = dataset.get_mini_dataset(
    eval_batch_size, eval_iters, shots, classes, split=True
)
print("test_images: ", test_images.shape, "test_labels: ",test_labels,"test_preds:", test_preds)
for images, labels in train_set:
    with tf.GradientTape() as tape:
        preds = model(images)
        loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
test_preds = model.predict(test_images)

##########################################改
import load_dataset as ld 

D = "E:\\data\\bottle\\train\\good\\"
E = "E:\\data\\bottle\\test\\broken_large\\"
B = "E:\\workspace\\project_\\bad_file\\"
S = "E:\\workspace\\project_\\file_smoke"
C = "E:\\workspace\\segmentation\\unt01_OK\\cat"
test_ldx=[]
test_ldy=[]
ld.d(D,test_ldx,test_ldy)
ld.d(E,test_ldx,test_ldy,1)
ld.d(B,test_ldx,test_ldy,2)
ld.d(S,test_ldx,test_ldy,3)
ld.d(C,test_ldx,test_ldy,4)
test_ldx=np.array(test_ldx)
test_ldy=np.array(test_ldy)

from sklearn.model_selection import train_test_split
test_ldx,X_test_img,test_ldy,y_test_label =  train_test_split(test_ldx, test_ldy,test_size=0.4,random_state=22 )
print("len(test_ldx) : ", len(test_ldx))
# test_ldx=np.array(test_ldx)
print("test_images",test_ldx.shape)
try:
    test_predsxx = model.predict(test_ldx[test_ldy==1])
    # test_predsxx= tf.argmax(test_predsxx).numpy()
    # print("test_predsxx: ",test_predsxx)
    def plot_image(image,labels,prediction,idx,num=10):  
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
    print(test_predsxx.argmax(axis=1))
    plot_image(test_ldx,test_ldy,test_predsxx.argmax(axis=1),idx=0)
except:
    pass



################################################
test_preds = tf.argmax(test_preds).numpy()

_, axarr = plt.subplots(nrows=1, ncols=5, figsize=(20, 20))

sample_keys = list(train_dataset.data.keys())

for i, ax in zip(range(len(test_labels)), axarr):
    
    #temp_image = np.stack((test_images[i, :, :, 0],) * 3, axis=2)
    temp_image = np.stack((test_images), )
    print("size: ", temp_image.shape)
    #temp_image = np.stack((test_images[i, :,:, 0],) , axis=2)
    temp_image *= 255
    temp_image = np.clip(temp_image, 0, 255).astype("uint8")
    ax.set_title(
        "Label : {}, Prediction : {}".format(int(test_labels[i]), test_preds[i])
    )
    ax.imshow(temp_image[i], cmap="gray")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
test_predsxx = model.predict(test_ldx)
test_predsxx=test_predsxx.argmax(axis=1)
try:
    print(f"accuracy_score: {accuracy_score(test_ldy, test_predsxx):.3f}")

    confusion = confusion_matrix(test_ldy, test_predsxx)

    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_matrix(test_ldy, test_predsxx), 
                cmap="Blues", annot=True, fmt="d", cbar=False,
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix")
    plt.show()
except:
    pass

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
try:
    import itertools
    import sklearn.metrics as metrics
    cnf_matrix = metrics.confusion_matrix(test_ldy, test_predsxx)#y_test_label
    target_names = ['OK', 'NG']
    plot_confusion_matrix(test_predsxx)
except:
    pass
