import numpy as np
import matplotlib.pyplot as plt
import pickle
from lib.hdplrmm import GibbsSampler
import lib.hdplrmm_original

###################################### generate simulation data #########################################
# 2, 2-dimensional Gaussian
mean = [[0., 0., 0.], [3., 3., 3.], [-5., -3., 1.]]
sigma_square = 0.01
seg = 5 # 5 users
directory = './img/'

color_iter = ['r', 'g', 'b', 'm', 'c']
marker_iter =['+', 'v', 'o', 's', '*']

num = [100, 50, 20, 30, 50]
dim = len(mean[0])
data_x = {index: np.zeros((num[index], dim)) for index in range(seg)} #np.zeros((seg, num, dim))
data_y = {index: np.zeros(num[index]) for index in range(seg)}  # np.zeros((seg, num))
true_lr_model = {index: np.zeros(num[index]) for index in range(seg)}
prob = np.random.dirichlet(np.ones(len(mean)), seg) # seq * len(mean)
for s in range(seg):
    for n in range(num[s]):
        tmp_c = np.random.choice(len(mean), 1, p=prob[s])[0]
        tmp_x = np.random.normal(0, 1, dim)
        l2_norm = np.linalg.norm(tmp_x, ord=2)
        tmp_x = tmp_x / l2_norm
        tmp_y = np.random.normal(np.array(mean[tmp_c]).transpose().dot(np.array(tmp_x)), np.sqrt(sigma_square))
        data_x[s][n, :] = tmp_x
        data_y[s][n] = tmp_y
        true_lr_model[s][n] = tmp_c

    plt.figure()
    X = data_x[s]
    print("Seg {} shape {}".format(s, X.shape))
    cluster = true_lr_model[s]
    for j in range(len(mean)):
        plt.scatter(X[np.array(cluster) == j, 0], X[np.array(cluster) == j, 1], marker=marker_iter[j], color=color_iter[j])
        plt.title('data distribution in time series %i' % (s+1))
    # pl.show()
    plt.savefig(directory + 'data_dist_%i' % s)


################################# add data point one by one ######################################
sampler = GibbsSampler(snapshot_interval=10, dimension=dim, compute_loglik=False)

sampler_original = lib.hdplrmm_original.GibbsSampler(snapshot_interval=10, compute_loglik=False)
sampler_original.initialize(data_x=data_x, data_y=data_y)
sampler_original.print_statistics()
print("===========================================================================================")
for userID in range(seg):
    X = data_x[userID]
    Y = data_y[userID]
    assert len(X) == len(Y)
    for data_index in range(X.shape[0]):
        sampler.add_data_point(userID, X[data_index], Y[data_index], [0, userID]) # table id = 0, dish id = user id
sampler.print_statistics()

snap_interval = 10
iteration = 10
for tmp in range(int(iteration/snap_interval)):
    sampler.sample(snap_interval)
    kdt = sampler._k_dt
    tdv = sampler._t_dv
    # pickle.dump(kdt, open(directory + 'kdt', 'wb'))
    # pickle.dump(tdv, open(directory + 'tdv', 'wb'))
    for s in range(seg):
        plt.figure()
        X = data_x[s]
        for j in range(sampler._K):
            plt.scatter(X[kdt[s][tdv[s]] == j, 0], X[kdt[s][tdv[s]] == j, 1], marker=marker_iter[j % 5], color=color_iter[j % 5])
            plt.title('data inference result in time series %i' % (s + 1))
        # plt.show()
        plt.savefig(directory + 'data_inference_%i' % s)

snap_interval = 10
iteration = 10
for tmp in range(int(iteration/snap_interval)):
    sampler_original.sample(snap_interval)
    kdt = sampler_original._k_dt
    tdv = sampler_original._t_dv
    # pickle.dump(kdt, open(directory + 'kdt', 'wb'))
    # pickle.dump(tdv, open(directory + 'tdv', 'wb'))
    for s in range(seg):
        plt.figure()
        X = data_x[s]
        for j in range(sampler_original._K):
            plt.scatter(X[kdt[s][tdv[s]] == j, 0], X[kdt[s][tdv[s]] == j, 1], marker=marker_iter[j % 5], color=color_iter[j % 5])
            plt.title('original implementation data inference result in time series %i' % (s + 1))
        # plt.show()
        plt.savefig(directory + 'original implementation data_inference_%i' % s)