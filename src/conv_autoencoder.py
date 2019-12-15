from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import cv2

def load_data(s):
    import glob
    import cv2
    x_files = glob.glob('/home/sid/thesis/dummy/ae_train/x/*.jpg')
    y_files = glob.glob('/home/sid/thesis/dummy/ae_train/y/*.jpg')
    shape = (len(x_files), s[0], s[1], s[2])
    x = np.empty(shape, dtype=np.uint16)
    y = np.empty(shape, dtype=np.uint16)
    for j, fname in enumerate(x_files):
        bgr = cv2.imread(fname)
        x[j] = cv2.resize(bgr, (shape[1], shape[2]))
    for j, fname in enumerate(y_files):
        bgr = cv2.imread(fname)
        y[j] = cv2.resize(bgr, (shape[1], shape[2]))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    return x_train, y_train, x_test, y_test

shape = (128, 128, 3)
input_img = Input(shape=(shape[0], shape[1], 3))  # adapt this if using `channels_first` image data format

x = Conv2D(512, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
plot_model(autoencoder, 'model.png')
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import cifar10
import numpy as np

x_train, y_train, x_test, y_test = load_data(shape) #cifar10.load_data()
#cv2.imshow("0", x_train[0])
c = 255.
x_train = x_train.astype('float32') / c
y_train = y_train.astype('float32') / c
x_test = x_test.astype('float32') / c
y_test = y_test.astype('float32') / c
#cv2.imshow("1", x_train[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

x_train = np.reshape(x_train, (len(x_train), shape[0], shape[1], shape[2]))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), shape[0], shape[1], shape[2]))
x_test = np.reshape(x_test, (len(x_test), shape[0], shape[1], shape[2]))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), shape[0], shape[1], shape[2]))

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, y_train,
                epochs=1,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(x_test)
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(shape[0], shape[1], shape[2]))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(shape[0], shape[1], shape[2]))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()