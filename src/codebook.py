
#create a codebook of 20,000 rotation, rotate them each 36 times randomly .
#render the views and train a neural network
import numpy as np

from renderer.pysixd_stuff import view_sampler
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
    i = 0
    for view in views:
        for cyclo in np.linspace(0, 2. * np.pi, num_cyclo):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i, :, :] = rot_z.dot(view['R'])
            i += 1
    return Rs

R = viewsphere_for_embedding()
print(len(R))

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, LSTM
from keras.models import Model

shape = (128, 128, 3)
input_img = Input(shape=(shape[0], shape[1], 3))
x = Conv2D(512, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

lstm = LSTM(64, return_sequences=False, input_shape)
