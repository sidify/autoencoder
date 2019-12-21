import numpy as np
import tensorflow as tf
from renderer.pysixd_stuff import view_sampler
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Flatten, TimeDistributed, BatchNormalization
from keras.models import Sequential


def __rot_matrix_to_euler_angles(R_batch):
    import math
    i = 0
    for R in R_batch:
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        euler_angles = np.array([x, y, z])
        print(euler_angles)
        i += 1
    return euler_angles

def viewsphere_for_embedding():
    num_cyclo = 36
    min_n_views = 20000
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
    #Rs = np.empty((len(views), 3, 3))
    i = 0
    for view in views:
        #Rs[i,:,:] = view['R']
        #i += 1

        for cyclo in np.linspace(0, 2. * np.pi, num_cyclo):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i, :, :] = rot_z.dot(view['R'])
            #print (Rs[i, :, :])
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
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='relu'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')


    Rs = viewsphere_for_embedding()
    #__rot_matrix_to_euler_angles(Rs)
    #np.take(Rs,np.random.permutation(Rs.shape[0]),axis=0,out=Rs)
    #Rs = Rs[0:1000]
    from sklearn.model_selection import train_test_split
    _, _, y_train, y_test = train_test_split(Rs, Rs, test_size=0.1)
    print(y_test.shape)
    from data_generator import DataGenerator
    from renderer import meshrenderer_phong as mp
    cad_model = '/home/sid/thesis/ply/models_cad/obj_05_red.ply'
    renderer = mp.Renderer(cad_model, samples=1, vertex_tmp_store_folder='.', clamp=False, vertex_scale=1.0)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.device('/cpu:0'):

        train_gen = DataGenerator(y_train, "train", renderer, save_rendered_img="/home/sid/thesis/dummy2/train/")
        test_gen = DataGenerator(y_test, "test", renderer, save_rendered_img="/home/sid/thesis/dummy2/test/")
        imgs_test, y_ = test_gen.data_gen(y_test[:1])
        model.fit_generator(generator=train_gen,
                            validation_data=test_gen,
                            epochs=1, workers=0
                            )

        y = model.predict(imgs_test)
        print ("Org val:", y_[0])
        print ("Predicted ", y[0])

        model.save("codebook_epochs_10.h5")

