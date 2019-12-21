from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Flatten, TimeDistributed, BatchNormalization
from keras.models import Sequential




if __name__ == "__main__" :
    #create a codebook of 20,000 rotation, rotate them each 36 times randomly .
    #render the views and train a neural network


    shape = (128,128,3)
    #input_img = Input(shape=(shape[0], shape[1], 3))
    cnn = Sequential()
    cnn.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(shape[0], shape[1], 3)))
    cnn.add(MaxPooling2D((2, 2), padding='same'))
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), padding='same'))
    cnn.add(Flatten())

    model = Sequential()
    #model.add(TimeDistributed(cnn))
    #model.add(LSTM(64))
    model.add(cnn)
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='relu'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')


    from data_generator2 import DataGenerator

    train_gen = DataGenerator(32, (128, 128, 3), data_path="/home/sid/thesis/dummy2/train/")
    test_gen = DataGenerator(32, (128, 128, 3), data_path="/home/sid/thesis/dummy2/test/")
    import tensorflow as tf
    with tf.device('/cpu:0'):
        model.fit_generator(generator=train_gen,
                        validation_data=test_gen,
                        epochs=1, workers=0
                        )


    model.save("codebook2_epochs_10.h5")

