import os
from scipy import misc
import numpy as np

def read_images_with_five_fold(path, size = None):
    print("Reading images.....")
    if size is not None:
        print ("Images will be resized by {}".format(str(size)))
    label_list = os.listdir(path)

    five_fold = [[1,3],[2,4],[5,7],[6,9],[8,10]]
    train_class = []
    test_class = []
    train_x = []
    test_x = []
    for fold in range(len(five_fold)):
        test_id = five_fold[fold]
        flag_1 = 0
        flag_2 = 0
        training_class_fold = []
        testing_class_fold = []
        for label in label_list:
            file_path =path + label
            for i in range(1,11):
                im = misc.imread(file_path + '/' + str(i) + '.pgm')
                if size is not None:
                    im = misc.imresize(im, size)
                im = im.reshape((1, im.shape[0]*im.shape[1]))
                if i in test_id:
                    if flag_1 == 0:
                        test_x_fold = np.copy(im)
                        flag_1 = 1
                        testing_class_fold.append(label)
                    else:
                        test_x_fold = np.vstack((test_x_fold, np.copy(im)))
                        testing_class_fold.append(label)
                else:
                    if flag_2 == 0:
                        train_x_fold = np.copy(im)
                        flag_2 = 1
                        training_class_fold.append(label)
                    else:
                        train_x_fold  = np.vstack((train_x_fold, np.copy(im)))
                        training_class_fold.append(label)
        train_class.append(training_class_fold)
        test_class.append(testing_class_fold)
        train_x.append(train_x_fold)
        test_x.append(test_x_fold)

    print('I equally divide the dataset to {} parts and seperately use four parts as train data, '
          'use the other one part as test data'.format(len(train_x)))
    print('one of train datas\' size is ({} * {})'.format(train_x[0].shape[0], train_x[0].shape[1]))
    print('one of test datas\' size is ({} * {})'.format(test_x[0].shape[0], test_x[0].shape[1]))

    return train_x, test_x, train_class, test_class




