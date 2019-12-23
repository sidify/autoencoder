import keras
import numpy as np
import cv2

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, dim, data_path, no_of_samples=5000):
        print("Data Path :", data_path)
        self.no_of_samples = no_of_samples
        self.batch_size = batch_size
        self.dim = dim
        self.data_path = data_path
        self.images_batch = np.zeros((batch_size, dim[0], dim[1], dim[2]), dtype=np.uint8)
        self.y = np.array(self.__csv_parser__(data_path))
        self.file_names = self.__read_files__(data_path) #keep only the same indexes as y
        self.indexes = np.arange(len(self.file_names))
        self.__shuffle__()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __shuffle__(self):
        s = np.arange(self.y.shape[0], dtype=int)
        np.random.shuffle((s))
        self.indexes = self.indexes[s]
        #find the largest multiple
        self.no_of_samples = int(self.no_of_samples/self.batch_size) * self.batch_size
        print("Using number of samples :", self.no_of_samples)
        self.indexes = self.indexes[0:self.no_of_samples]
        #self.y = self.y[0:self.no_of_samples]

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
        files_dict = {}
        import glob
        path = path + 'x/*.jpg'
        paths = glob.glob(path)
        for path in paths:
            loc1 = path.rfind('/')
            loc2 = path.rfind('.')
            files_dict[int(path[loc1+1 : loc2])] = path
        return files_dict


    def __data_generation__(self, files):
        i = 0
        for file in files:
            img = cv2.imread(file)
            if img.shape != self.dim:
                print("dim of input image not same as initialised, so resizing...")
                img = cv2.resize(img,(self.dim[0], self.dim[1], self.dim[2]))
            self.images_batch[i] = img
            i += 1
        return self.images_batch,

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        files = [self.file_names[k] for k in indexes]
        # Generate data
        self.__data_generation__(files)
        k = self.y[indexes]
        return self.images_batch, self.y[indexes]
