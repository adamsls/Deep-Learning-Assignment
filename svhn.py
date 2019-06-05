import os
from six.moves.urllib.request import urlretrieve
import sys
import tarfile
import numpy as np
from PIL import Image
from keras.layers import Activation, Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.models import Model
from keras.models import load_model
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io 


def test(filename):
    
    img = Image.open(filename)

    basewidth = 32
    baseheight = 32
    
    img = img.resize((basewidth, baseheight), Image.ANTIALIAS)
    #img.save('resized_image.png')
    #
    #new_img = Image.open('resized_image.png')
    new_img_array = np.asarray(img)
    scalar = 1 / 255.
    new_img_array_norm = new_img_array * scalar
    
    test_array = np.ndarray(shape=(1, 32, 32, 3), dtype='float32')
    test_array[0, :, :, :] = new_img_array_norm
     
    model_test = load_model('model3.h5')
    new_prediction = model_test.predict((test_array))
    
    return np.argmax(new_prediction)

#function to download datasets, train model and test
#https://github.com/hangyao/street_view_house_numbers/blob/master/3_preprocess_multi.ipynb
    
def traintest():
    
    url = 'http://ufldl.stanford.edu/housenumbers/'
    
    def download(filename, force=False):
      if force or not os.path.exists(filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, filename)
        print('\nDownload Complete!')
      return filename
    
    train_filename = download('train_32x32.mat')
    test_filename = download('test_32x32.mat')
    
    #load svhn training data
    train_mat = scipy.io.loadmat('train_32x32.mat')
    test_mat = scipy.io.loadmat('test_32x32.mat')
    
    #group training data into input and output
    X_train_mat = train_mat['X']
    X_train_shaped = np.moveaxis(X_train_mat, -1, 0)
    y_train = train_mat['y']
    
    #view sample images
    plt.imshow(X_train_shaped[5])
    print(y_train[5])
       
    #group testing data into input and output
    X_test_mat = test_mat['X']
    X_test_shaped = np.moveaxis(X_test_mat, -1, 0)
    y_test = test_mat['y']   
    plt.imshow(X_test_shaped[16])
    print(y_test[16])

    
    #***************************************************
    
    #Normalise etc
    #https://github.com/tohinz/SVHN-Classifier/blob/master/preprocess_svhn.py
    
    # replace label "10" with label "0"
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    
    # normalize images so pixel values are in range [0,1]
    scalar = 1 / 255.
    X_test = X_test_shaped * scalar
    X_train = X_train_shaped * scalar
           
    #https://ryannng.github.io/2016/12/20/Street-View-House-Numbers-Recognition-Using-ConvNets/
  
    def build_model():
        input_ = Input(shape=(32, 32, 3))
    
        # conv layer 1
        model = BatchNormalization()(input_)
        model = Conv2D(64, (7, 7), activation ='relu', padding='same')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
    
        # conv layer 2
        model = BatchNormalization()(model)
        model = Conv2D(128, (5, 5), activation ='relu', padding='valid')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
    
        # conv layer 3
        model = BatchNormalization()(model)
        model = Conv2D(256, (3, 3), activation ='relu', padding='valid')(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Dropout(0.5)(model)
    
        # fully connected layers
        model = Flatten()(model)
        model = Dense(1024, activation='relu')(model)
        model = Dense(512, activation='relu')(model)
    
        x = Dense(10,  activation='softmax')(model)
        
        model = Model(inputs=input_, outputs=x)
        return model
    
    model = build_model()
    model.summary()        
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="rmsprop",
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=64, epochs=20)
    
    result = model.evaluate(X_test, y_test)
    
    return result

