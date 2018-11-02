import pickle
from Data_process import read_images_with_five_fold
from SVM import SVM
from PCA import PCA
from collections import Counter
import numpy as np

def get_data(path):
    train_x, test_x, train_class, test_class = read_images_with_five_fold(path)

    return train_x, test_x, train_class, test_class

def pca_data(train_x, test_x, fold):
    pca = PCA()
    train_x, test_x = pca.process(train_x, test_x)
    #with open('train_pca_'+str(fold)+'.pkl', 'wb') as f1:
        #pickle.dump(train_x, f1)
    #with open('test_pca_'+str(fold)+'.pkl', 'wb') as f2:
        #pickle.dump(test_x, f2)

    #with open('train_pca_'+str(fold)+'.pkl', 'rb') as f1:
        #train_x = pickle.load(f1)
    #with open('test_pca_'+str(fold)+'.pkl', 'rb') as f2:
        #test_x = pickle.load(f2)
    train_x = train_x.astype(np.float, copy=True)
    test_x = test_x.astype(np.float, copy=True)
    return train_x, test_x

def train_and_test(train_x, test_x, training_class, testing_class, kernel):
    classes = Counter(training_class)
    classes = classes.keys()
    total_accuracy = []
    for label in classes:
        train_y = []
        for t in training_class:
            if t == label:
                train_y.append(1.0)
            else:
                train_y.append(-1.0)
        train_y = np.array(train_y)

        test_y = []
        for t in testing_class:
            if t == label:
                test_y.append(1.0)
            else:
                test_y.append(-1.0)
        test_y = np.array(test_y)

        classfier = SVM(kernel=kernel, C=0.1)
        classfier.train(train_x, train_y)
        y_predict = classfier.test(test_x)
        correct = np.sum(y_predict == test_y)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        accuracy = correct / len(y_predict)
        print("accuracy is {}".format(accuracy))
        total_accuracy.append(accuracy)
    mean_accuracy = np.mean(np.array(total_accuracy))
    print('mean accuracy is {}'.format(mean_accuracy))

    return mean_accuracy

if __name__ == '__main__':
    path = 'att_faces/'
    train_x, test_x, train_class, test_class = get_data(path)
    fold_total_mean_accuracy = []
    kernels=['linear', 'polynomial2', 'polynomial3', 'polynomial4', 'polynomial8', 'gaussian']
    for fold in range(5):
        train_x_fold = train_x[fold]
        test_x_fold = test_x[fold]
        train_class_fold = train_class[fold]
        test_class_fold = test_class[fold]

        train_x_fold = train_x_fold.astype(np.float, copy=True)
        test_x_fold = test_x_fold.astype(np.float, copy=True)

        train_x_fold, test_x_fold = pca_data(train_x_fold, test_x_fold, fold)
        mean_accuracy = train_and_test(train_x_fold, test_x_fold, train_class_fold, test_class_fold, kernels[0])
        print('the mean of accuracy of 40 SVM classifiers for {1} fold is {0}'.format(mean_accuracy, str(fold+1)))
        fold_total_mean_accuracy.append(mean_accuracy)
    all_fold_mean_accuracy = np.mean(np.array(fold_total_mean_accuracy))
    print('the mean of accuracy of 40 SVM classifiers is {}'.format(all_fold_mean_accuracy))



