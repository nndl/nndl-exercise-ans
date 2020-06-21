import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_data(dot_num =200):
    x_1_p = np.random.normal(3.,1,dot_num)
    y_1_p = np.random.normal(8.,1,dot_num)
    C1 = np.array([x_1_p,y_1_p]).T

    x_2_p = np.random.normal(8.,1,dot_num)
    y_2_p = np.random.normal(3.,1,dot_num)
    C2 = np.array([x_2_p,y_2_p]).T

    x_3_p = np.random.normal(9.,1,dot_num)
    y_3_p = np.random.normal(9.,1,dot_num)
    C3 = np.array([x_3_p,y_3_p]).T

    res = np.concatenate((C1,C2,C3),axis=0)
    plt.scatter(res[:,0],res[:,1],c='b')

    plt.show()
    return res

class GMM:
    def __init__(self,k,d):
        """
        this function is init gmm function
        :param k: the number of target
        :param d: the number of prob
        """
        self.k = k
        self.p = np.random.rand(k)
        self.p = self.p/self.p.sum()

        self.means = np.random.rand(k,d)
        self.cov = np.empty((k,d,d))
        for i in range(k):
            self.cov[i] = np.eye(d)*np.random.rand(1) * k


    def fit(self,data):
        for _ in range(1000):
            density = np.empty((len(data),self.k))
            for i in range(self.k):
                norm = stats.multivariate_normal(self.means[i],self.cov[i])
                density[:,i] = norm.pdf(data)
            # print(density)
            postporb = density*self.p
            postporb = postporb/postporb.sum(axis=1,keepdims=True)

            p_new = postporb.sum(axis=0)
            mean_new = np.tensordot(postporb,data,axes=[0,0])

            cov_new = np.empty(self.cov.shape)
            for i in range(self.k):
                tmp = data-self.means[i]
                cov_new[i] = np.dot(tmp.T*postporb[:,i],tmp)/p_new[i]

            self.cov = cov_new
            self.p = p_new/len(data)
            self.means = mean_new / p_new.reshape(-1,1)

        return self.means

    def get_distense(self, point_data):
        res = []
        for mean in self.means:
            res.append(np.linalg.norm(point_data-mean))

        return np.array(res)

    def predict(self, data):
        res = []

        for i in range(data.shape[0]):
            dis = self.get_distense(data[i])

            res.append(dis.argmin(axis=-1))

        return res



data = get_data()

a = GMM(3,2)
a.fit(data)
y = a.predict(data)

color = ['r','b','g']
y_c = [color[i] for i in y]

plt.scatter(data[:, 0], data[:, 1], c=y_c)
plt.show()