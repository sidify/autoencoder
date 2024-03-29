import numpy as np
import keras
from sklearn.model_selection import train_test_split
import cv2
from renderer.pysixd_stuff import view_sampler
import math

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, list_IDs, type, renderer, batch_size=32, dim=(32,128, 128, 3), shuffle=True, save_rendered_img = None):
        print ("Init class :" + type)
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.rendered = np.empty((batch_size, 128, 128, 3), dtype=np.uint8)
        self.euler_angles = np.empty((batch_size, 3))
        self.renderer = renderer
        self.save_rendered_img = save_rendered_img
        self.count_rendered_img = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def data_gen(self, R_batch):
        return self.__data_generation(R_batch)

    def __data_generation(self, R_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.__render_for_rotations(R_batch, self.renderer)
        y = self.__rot_matrices_to_euler_angles(R_batch)
        return X, y

    def __render_for_rotations(self, R, renderer):
        height = 720
        width = 960
        clip_near = 10
        clip_far = 1000

        #rendered = np.empty((len(R), 128, 128, 3), dtype=np.uint8)
        K = np.array(([1029.87472, 0, 480, 0, 1029.69249, 350, 0, 0, 1])).reshape((3, 3))
        t = np.array(([0, 0, 700]), dtype=np.float16)
        i = 0
        import csv
        csv_file = str(self.save_rendered_img) + "rot_labels.csv"
        with open(csv_file, mode = 'a+') as csv_writer:
            writer = csv.writer(csv_writer, delimiter=',')

        #print(hex(id(self.renderer)))
            for rot in R:
                color, depth_x = renderer.render(
                    0, int(width), int(height), K
                    , rot, t, clip_near, clip_far)
                ys, xs = np.nonzero(depth_x > 0)
                ys = np.array(ys, dtype=np.int16)
                xs = np.array(xs, dtype=np.int16)
                x, y, w, h = view_sampler.calc_2d_bbox(xs, ys, (width, height))
                img = color[y:y + h, x:x + w]
                img = cv2.resize(img, (128, 128))
                self.rendered[i, :, :] = img
                if self.save_rendered_img != None :
                    eu = self.__rot_matrix_to_euler_angle(rot)
                    writer.writerow([self.count_rendered_img, eu[0], eu[1], eu[2]])
                    file_name = self.save_rendered_img + "x/" + str(self.count_rendered_img) + ".jpg"
                    cv2.imwrite(file_name, img)
                    self.count_rendered_img += 1
                i += 1
        return self.rendered

    def __rot_matrix_to_euler_angle(self, R):

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

        return np.array([x, y, z])

    def __rot_matrices_to_euler_angles(self, R_batch):
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

            self.euler_angles[i] = np.array([x, y, z])
            i += 1
        return self.euler_angles