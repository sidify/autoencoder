import numpy as np
import tensorflow as tf
from renderer.pysixd_stuff import view_sampler
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Flatten, TimeDistributed, BatchNormalization
from keras.models import Sequential


def viewsphere_for_embedding():
    num_cyclo = 36
    min_n_views = 200
    radius = 700
    azimuth_range = (0, 2 * np.pi)
    elev_range = (-0.5 * np.pi, 0.5 * np.pi)
    views, _ = view_sampler.sample_views(
        min_n_views,
        radius,
        azimuth_range,
        elev_range
    )
    Rs = np.empty((len(views) * num_cyclo, 3, 3))
    i = 0
    for view in views:
        for cyclo in np.linspace(0, 2. * np.pi, num_cyclo):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i, :, :] = rot_z.dot(view['R'])
            # print (Rs[i, :, :])
            i += 1
    return Rs

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
    model.add(Dense(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='relu'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')


    Rs = viewsphere_for_embedding()
    #np.take(Rs,np.random.permutation(Rs.shape[0]),axis=0,out=Rs)
    #Rs = Rs[0:1000]
    from sklearn.model_selection import train_test_split
    _, _, y_train, y_test = train_test_split(Rs, Rs, test_size=0.1)

    from data_generator import DataGenerator
    from renderer import meshrenderer_phong as mp
    cad_model = '/home/sid/thesis/ply/models_cad/obj_05_red.ply'
    renderer = mp.Renderer(cad_model, samples=1, vertex_tmp_store_folder='.', clamp=False, vertex_scale=1.0)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    with tf.device('/gpu:0'):

        train_gen = DataGenerator(y_train, "train", renderer)
        test_gen = DataGenerator(y_test, "test", renderer)
        imgs_test, y_ = test_gen.data_gen(y_test[:1])
        model.fit_generator(generator=train_gen,
                            validation_data=test_gen,
                            epochs=3, workers=0
                            )

        print ("Org val:", y_[0])
        print ("Predicted ", y[0])





