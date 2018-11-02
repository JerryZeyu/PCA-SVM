# PCA-SVM
## Dataset
The dataset consists of 40 classes and each class have 10 pictures.
## Usage
Download the data
```bash
# download data
$ sh download.sh
```
Run
```bash
$ python main.py 
```
In order to improve the performance, I implemented the PCA, because I don't use any library for it, the time for Eigendecomposition is very long. So you can choose to dump them when you firstly run them. 

For the SVM, I have implemented different kernels, if you would like to use different kernels, you can assign the index of kernels in the main.py.

In order to avoid the overfitting, I also implemented the 5 fold cross validation.
