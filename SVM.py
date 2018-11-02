import numpy as np
import cvxopt
import cvxopt.solvers
from numpy import linalg


class SVM(object):
    def __init__(self, kernel=None, C=None):
        self.kernel = kernel
        self.C = C
        if kernel == 'linear':
            self.const = 1e-8
        elif kernel == 'polynomial2':
            self.const = 1e-15
        elif kernel == 'polynomial3':
            self.const = 1e-22
        elif kernel == 'polynomial4':
            self.const = 1e-29
        elif kernel == 'polynomial8':
            self.const = 1e-58
        elif kernel == 'gaussian':
            self.const = 1e-3
        if self.C is not None:
            self.C = float(self.C)

    def kernel_func(self, x, y, kernel):
        if kernel == 'polynomial2':
            return (1 + np.dot(x, y)) ** 2
        if kernel == 'polynomial3':
            return (1 + np.dot(x, y)) ** 3
        if kernel == 'polynomial4':
            return (1 + np.dot(x, y)) ** 4
        if kernel == 'polynomial8':
            return (1 + np.dot(x, y)) ** 8
        if kernel == 'linear':
            return np.dot(x, y)
        if kernel == 'gaussian':
            return np.exp(-linalg.norm(x - y) ** 2 / (2 * (5.0 ** 2)))

    def train(self, x, y):
        samples_num, features_num = x.shape

        self.K = np.zeros((samples_num, samples_num))
        for i in range(samples_num):
            for j in range(samples_num):
                self.K[i, j] = self.kernel_func(x[i], x[j], self.kernel)

        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(np.ones(samples_num) * -1)
        A = cvxopt.matrix(y, (1, samples_num))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(samples_num) * -1))
            h = cvxopt.matrix(np.zeros(samples_num))
        else:
            tmp1 = np.diag(np.ones(samples_num) * -1)
            tmp2 = np.identity(samples_num)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(samples_num)
            tmp2 = np.ones(samples_num) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])
        #print('a: ',a)
        sv = a > self.const
        #print('sv: ',sv)
        ind = np.arange(len(a))[sv]
        # print('ind: ',ind)
        self.a = a[sv]
        # print('self.a: ',self.a)
        self.sv = x[sv]
        self.sv_y = y[sv]

        self.b = self.get_intercept(sv, ind)
        self.w = self.get_weightvector(features_num)

    def get_intercept(self, sv, ind):
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * self.K[ind[n], sv])
        self.b /= len(self.a)

        return self.b

    def get_weightvector(self,features_num):
        if self.kernel == 'linear':
            self.w = np.zeros(features_num)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def test(self, x):
        if self.w is not None:
            return np.dot(x, self.w) + self.b
        else:
            y_predict = np.zeros(len(x))
            for i in range(len(x)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel_func(x[i], sv, self.kernel)
                y_predict[i] = s
        return np.sign(y_predict + self.b)