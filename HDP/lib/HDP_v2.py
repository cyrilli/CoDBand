# -*- coding: utf-8 -*-
"""
@author: cl5ev
"""
import numpy as np

from hdplrmm import GibbsSampler
import random

class HDPMBandit:
    def __init__(self, dimension, alpha=0.6):
        self.alpha = alpha
        self.d = dimension

        self.tau = 100 # look at a user's recent tau data when selecting a model

        self.sampler = GibbsSampler(snapshot_interval=10, dimension=dimension, compute_loglik=False)

        self.table_topic_for_selected_arm = []

        self.verbose = True
        self.CanEstimateUserPreference = True

        self.frequency_of_gibbs_sampling = 100  # how often do we do gibbs sampling
        self.gibbs_iter_num = 5  # number of iterations of gibbs sampling
        self.step_count = 0
        self.previous_estimated_theta = {}

    def decide(self, pool_articles, userID):

        # Model Selection
        # be careful with the case when sampler has no data
        # and when sampler has no data for this user
        table_index, topic_index, model = self.sampler.select_model(userID, self.tau)
        self.table_topic_for_selected_arm = [table_index, topic_index]
        # ARM SELECTION
        maxUCB = float('-inf')
        articlePicked = None
        if self.verbose:
            print("SELECTED for user {}: table_index {}, topic_index {}".format(userID, table_index,topic_index))
        for x in pool_articles:
            x_ucb = self.calc_UCB(model, x.contextFeatureVector[:self.d])
            # pick article with highest Prob
            if maxUCB < x_ucb:
                articlePicked = x
                maxUCB = x_ucb
        self.previous_estimated_theta[userID] = model.mean.copy()
        return articlePicked

    def calc_UCB(self, model, article_FeatureVector):
        return model.get_UCB(article_FeatureVector, self.alpha)

    def updateParameters(self, articlePicked, reward, userID):
        self.step_count += 1
        # add new data point into sampler
        self.sampler.add_data_point(userID, articlePicked.featureVector, reward, self.table_topic_for_selected_arm)

        if self.step_count % self.frequency_of_gibbs_sampling == 0:
            print("step {}".format(self.step_count))
            self.GibssSampling()
        print("Number of models: {}".format(self.sampler._K))

    def GibssSampling(self):
        print("Apply Gibbs Sampling!")
        self.sampler.sample(self.gibbs_iter_num)

    def getTheta(self, userID):
        return self.previous_estimated_theta[userID]

if __name__ == '__main__':
    dim = 20
    hdp_bandit = HDPMBandit(dim)

    sigma_square = 0.01
    num_user = 12
    num_theta = 3
    # set of all possible user preferences
    # mean = [[0., 0., 1.], [3., 3., 3.], [-5., -3., 1.]]
    # !!! I noticed that we need to generate the mean so that they more different enough
    # if we generate mean in range (0,1), the result will be bad
    mean = np.random.random_integers(low=-10, high=10, size=(num_theta, dim)).tolist()
    prob = np.random.dirichlet(np.ones(len(mean)), num_user)  # seq * len(mean)

    correct_model_assignment = {}
    inferenced_model_assignment = {}
    for u in range(num_user):
        correct_model_assignment[u] = []
        inferenced_model_assignment[u] = []

    num_iter = 40 * num_user
    userID = 0
    for iter in range(num_iter):
        # table_index, topic_index, model = hdp_bandit.sampler.select_model(userID=userID, tau=hdp_bandit.tau)
        # print("SELECTED for user {}: table_index {}, topic_index {}".format(userID, table_index,topic_index))
        # assume we have selected an arm and received a reward
        # now we add new data point
        # generate a vector and a reward
        context_vector = np.random.normal(0, 1, dim)
        context_vector = context_vector / np.linalg.norm(context_vector, ord=2)
        tmp_c = np.random.choice(len(mean), 1, p=prob[userID])[0]
        correct_model_assignment[userID].append(tmp_c)
        user_theta = np.array(mean[tmp_c])
        user_theta = user_theta / np.linalg.norm(user_theta, ord=2)
        reward = np.random.normal(user_theta.transpose().dot(np.array(context_vector)), np.sqrt(sigma_square))

        hdp_bandit.sampler.add_data_point(userID=userID, x=context_vector, y=reward, table_topic_for_selected_arm=[0, userID])
        hdp_bandit.sampler.print_statistics()

        userID += 1
        if userID == num_user:
            userID = 0

    hdp_bandit.GibssSampling()
    kdt = hdp_bandit.sampler._k_dt
    tdv = hdp_bandit.sampler._t_dv
    for u in range(num_user):
        inferenced_model_assignment[u] = kdt[u][tdv[u].tolist()].astype(int).tolist()
    print(correct_model_assignment)
    print(inferenced_model_assignment)