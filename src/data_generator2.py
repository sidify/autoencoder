import keras
import numpy as np
import cv2

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, dim, data_path):
        print("Data Path :", data_path)
        self.batch_size = batch_size
        self.dim = dim
        self.data_path = data_path
        self.images_batch = np.empty((batch_size, dim[0], dim[1], dim[2]), dtype=np.uint8)
        self.y = self.__csv_parser__(data_path)
        self.file_names = self.__read_files__(data_path)
        self.indexes = np.arange(len(self.file_names))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __csv_parser__(self, path):
        file = path + "/y/rot_labels.csv"
        num_lines = sum(1 for line in open(file))
        y = np.empty((num_lines, 3), dtype=np.float16)
        import csv
        with open(file, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                #print(row[0], row[1], row[2], row[3])
                y[int(row[0])] = np.array((row[1], row[2], row[3]), dtype=np.float16)
        return y


    def __read_files__(self, path):
        import glob
        path = path + 'x/*.jpg'
        return glob.glob(path)

    def __data_generation__(self, files):
        i = 0
        for file in files:
            self.images_batch[i] = cv2.imread(file)
            i += 1
        return self.images_batch,

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        files = [self.file_names[k] for k in indexes]
        # Generate data
        self.__data_generation__(files)
        return self.images_batch, self.y[indexes]
