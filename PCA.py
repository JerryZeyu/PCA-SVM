import numpy as np

class PCA(object):
    def __init__(self):
        super(PCA).__init__()

    def eigen(self, data, shape= None):
        print("Eigendecomposition Running....")
        dim_mean = data.mean(axis = 0)
        print("dim_mean.shape: ", dim_mean.shape)
        if shape is not None:
            dis_im_mean = dim_mean.reshape(shape)
        reshaped_mean = dim_mean.reshape((1, dim_mean.shape[0]))
        print("reshaped_dim_mean.shape", reshaped_mean.shape)
        print("Subtracting mean from matrix data ...")
        data_ = data - reshaped_mean
        data_T = data_.transpose()
        print("Transposing matrix data, the shape will be {0}x{1} ".format(data_T.shape[0],data_T.shape[1]))

        mult = np.dot(data_T, data_)/float(data.shape[1] - 1)
        print("Computing coveriance matrix (data^T * data)/({0} - 1), the shape would be {1}x{2} ".format(data.shape[1],mult.shape[0],mult.shape[1]))
        e_values, e_vectors = np.linalg.eigh(mult)
        #print(len(e_vectors))
        #print(len(e_values))
        idx = np.argsort(-e_values)
        e_values = e_values[idx]
        e_vectors = e_vectors[:,idx]
        #print(e_vectors[0])
        print("Computing eigenvalues and eigenvectors and sort them descendingly, eigenvectors is {0}x{1} ".format(e_vectors.shape[0],e_vectors.shape[1]))

        return e_values, e_vectors

    def process(self, train, test, shape=None):
        dim = 100
        e_values_train, e_vectors_train = self.eigen(train, shape)
        dim_mean_train = train.mean(axis=0)
        reshaped_mean_train = dim_mean_train.reshape((1, dim_mean_train.shape[0]))
        train_ = train - reshaped_mean_train
        p_train = e_vectors_train[:, 0:dim].copy()
        train = np.dot(train_, p_train)

        e_values_test, e_vectors_test = self.eigen(test, shape)
        dim_mean_test = test.mean(axis=0)
        reshaped_mean_test = dim_mean_test.reshape((1, dim_mean_test.shape[0]))
        test_ = test - reshaped_mean_test
        p_test = e_vectors_test[:, 0:dim].copy()
        test = np.dot(test_, p_test)

        return train, test