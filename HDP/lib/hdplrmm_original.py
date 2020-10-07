"""
HDP Linear Regression Mixture model

Chuanhao Li
"""

import numpy as np
import math
import pickle
import scipy.special as ssp
from scipy.special import gammaln
# from model_loglikelihood import dict2mix, all_loglike
np.seterr(divide='ignore')
import pdb, traceback, sys


class Gaussian:
    def __init__(self, X, Y, mu_0=None, K_0=None, sigma_square=None):
        assert len(X) == len(Y)

        self.n = X.shape[0]
        self.dim = X.shape[1]

        if mu_0 is None:  # prior mean of theta
            self._mu_0 = np.zeros((self.dim,))
        else:
            self._mu_0 = mu_0
        assert(self._mu_0.shape == (self.dim,))

        if K_0 is None:  # prior variance of theta
            self._K_0 = 100.0 * np.eye(self.dim)
        else:
            self._K_0 = K_0
        assert(self._K_0.shape == (self.dim, self.dim))

        if sigma_square is None:  # prior variance of theta
            self._sigma_square = 0.01
        else:
            self._sigma_square = sigma_square

        self.mean = None
        self.covar = None
        if X.shape[0] > 0:
            self.fit(X, Y)
        else:
            self.default()

    def default(self):
        self.mean = self._mu_0
        self.covar = self._K_0
        np.linalg.cholesky(self.covar)

    def recompute_ss(self):
        self.n = self._X.shape[0]
        self.dim = self._X.shape[1]
        if self.n <= 0:
            self.default()
            return
        # update conjugate posterior parameters
        self.covar = np.linalg.inv(np.linalg.inv(self._K_0) + self._sum_xxT / self._sigma_square)
        np.linalg.cholesky(np.linalg.inv(self._K_0) + self._sum_xxT / self._sigma_square)
        np.linalg.cholesky(self.covar)
        self.mean = self.covar.dot(np.linalg.inv(self._K_0).dot(self._mu_0) + self._sum_xy / self._sigma_square)
        assert(self.covar.shape == (self.dim, self.dim))
        assert(self.mean.shape == (self.dim, ))

    def fit(self, X, Y):  # fit data
        self._X = X
        self._Y = Y

        self._sum_xxT = X.transpose().dot(X)
        self._sum_xy = X.transpose().dot(Y)
        assert(self._sum_xxT.shape == (self.dim, self.dim))
        assert(np.all(np.linalg.eigvals(self._sum_xxT) >= -1e-8))
        assert (self._sum_xy.shape == (self.dim, ))
        self.recompute_ss()

    def rm_point(self, x, y):
        """
        remove a point to current Gaussian mixture
        @:param x: data point to be removed
        """
        assert (self._X.shape[0] > 0)
        # Find the index of the point x in self._X
        indices = (abs(self._X - x)).argmin(axis=0)
        indices = np.matrix(indices)
        ind = indices[0, 0]
        for ii in indices:
            if (ii - ii[0] == np.zeros(len(ii))).all():
                ind = ii[0, 0]
                break

        self._X = np.delete(self._X, ind, axis=0)
        self._Y = np.delete(self._Y, ind, axis=0)
        self.n -= 1
        assert(self._X.shape == (self.n, self.dim))
        assert (self._Y.shape == (self.n, ))
        # self._sum_xxT -= np.outer(x,x)
        # self._sum_xy -= x*y
        self._sum_xxT = self._X.transpose().dot(self._X)
        self._sum_xy = self._X.transpose().dot(self._Y)
        assert(self._sum_xxT.shape == (self.dim, self.dim))
        assert (np.all(np.linalg.eigvals(self._sum_xxT) >= -1e-8))
        assert (self._sum_xy.shape == (self.dim, ))
        self.recompute_ss()

    def add_point(self, x, y):
        """
        add a point from current Gaussian mixture
        @:params x: data point to be added
        """
        if self.n <= 0:
            self._X = np.reshape(x, [1, self.dim])
            self._Y = np.reshape(y, [1,])
            self.n += 1
            assert (self._X.shape == (self.n, self.dim))
            assert (self._Y.shape == (self.n,))
            self._sum_xxT = np.outer(x,x)
            self._sum_xy = x.dot(y)
            assert (self._sum_xxT.shape == (self.dim, self.dim))
            # print(np.linalg.eigvals(self._sum_xxT))
            assert (np.all(np.linalg.eigvals(self._sum_xxT) >= -1e-8))
            assert (self._sum_xy.shape == (self.dim,))

        else:

            self._X = np.append(self._X, [x], axis=0)
            self._Y = np.append(self._Y, [y], axis=0)
            self.n += 1
            assert (self._X.shape == (self.n, self.dim))
            assert (self._Y.shape == (self.n,))
            self._sum_xxT = self._X.transpose().dot(self._X)
            self._sum_xy = self._X.transpose().dot(self._Y)
            assert (self._sum_xxT.shape == (self.dim, self.dim))
            assert (self._sum_xy.shape == (self.dim,))
        self.recompute_ss()

    def pdf(self, x, y):    # compute the log prob density of data point x
        size = len(x)
        assert size == self.mean.shape[0]
        assert (size, size) == self.covar.shape

        var = self._sigma_square + x.transpose().dot(self.covar).dot(x)
        mean = x.transpose().dot(self.mean)
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(y) - float(mean)) ** 2 / (2 * var))
        return num / denom

    def predictive_logpdf(self, x, y):    # compute the log prob density of data point x
        size = len(x)
        assert size == self.mean.shape[0]
        assert (size, size) == self.covar.shape

        var = self._sigma_square + np.matmul(np.matmul(x.transpose(), self.covar), x)
        mean = x.transpose().dot(self.mean)
        norm_const = -0.5 * np.log((2 * math.pi * var))

        res = -(float(y) - float(mean)) ** 2 / (2 * float(var))

        return norm_const + res


class GibbsSampler(object):
    """
    @:param snapshot_interval: the interval for exporting a snapshot of the model
    """
    def __init__(self, snapshot_interval=1, compute_loglik=False):
        self._snapshot_interval = snapshot_interval
        self._flag_compute_loglik = compute_loglik

        self._table_info_title = "Table-information-"
        self._dish_info_title = "Dish-information-"
        self._hyper_parameter_title = "Hyper-parameter-"

        self.params = {}

        # initialize the parameters of gamma prior of alpha
        self._alpha_a = 10.
        self._alpha_b = 1.

        self._gamma_a = 1.
        self._gamma_b = 1.

        self._alpha = np.random.gamma(self._alpha_a, self._alpha_b)
        self._gamma = np.random.gamma(self._gamma_a, self._gamma_b)

        self._D = 0
        self._K = 0

    """
    @param data: a N-by-D np array object, defines N points of D dimension
    @param K: number of topics, number of broke sticks
    @param gamma: the smoothing value for a table to be assigned to a new topic
    @param alpha: the smoothing value for a word to be assigned to a new table

    """
    def initialize(self, data_x, data_y):
        # initialize the documents
        self._corpus_x = data_x
        self._corpus_y = data_y
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._corpus_x)

        # initialize the total number of topics.
        self._K = self._D



        # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
        # self._n_kv = np.zeros((self._K, self._V))
        # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        self._n_kd = np.zeros((self._K, self._D)).astype(int)
        # initialize the table count matrix indexed by topic id, i.e., m_{\cdot k}
        self._m_k = np.zeros(self._K)

        # initialize the table information vectors indexed by document id and word id, i.e., t{j i}
        self._t_dv = {}
        # initialize the topic information vectors indexed by document id and table id, i.e., k_{j t}
        self._k_dt = {}
        # initialize the word count vectors indexed by document id and table id, i.e., n_{j t \cdot}
        self._n_dt = {}

        # we assume all words in a document belong to one table which is assigned to topic d, where d is the index of
        # document
        for d in range(self._D):
            # initialize the table information vector indexed by document and records down which table a word belongs to
            self._t_dv[d] = np.zeros(len(self._corpus_x[d]), dtype=np.int).astype(int)

            # self._k_dt records down which topic a table was assigned to
            self._k_dt[d] = np.array([d]).astype(int)
            assert(len(self._k_dt[d]) == len(np.unique(self._t_dv[d])))

            # word_count_table records down the number of words sit on every table
            self._n_dt[d] = np.zeros(1, dtype=np.int) + len(self._corpus_x[d])
            assert(len(self._n_dt[d]) == len(np.unique(self._t_dv[d])))
            assert(np.sum(self._n_dt[d]) == len(self._corpus_x[d]))

            self._n_kd[d, d] = len(self._corpus_x[d])

            # here, document index equals topic index
            self._m_k[d] = 1
            # initialize Gaussian mixtures, each document belongs to one cluster
            self.params[d] = Gaussian(X=self._corpus_x[d], Y = self._corpus_y[d])

    def print_statistics(self):
        print("===Number of users: {}, number of topics: {}".format(self._D, self._K))
        print("===Corpus x")
        for userID, data in self._corpus_x.items():
            print("user {} : data shape {}".format(userID, data.shape))
        print("===Corpus y")
        for userID, data in self._corpus_y.items():
            print("user {} : data shape {}".format(userID, data.shape))
        print("===Counts of data points [user, table] n_dt")
        for userID, data in self._n_dt.items():
            for table in range(len(data)):
                print("user {} : on table {}: num of data points {}".format(userID, table, data[table]))
        print("===Index of model [user, table] k_dt")
        for userID, data in self._k_dt.items():
            for table in range(len(data)):
                print("user {} : on table {}: index of model {}".format(userID, table, data[table]))
        print("===Index of table [user, data] t_dv")
        for userID, data in self._t_dv.items():
            for v in range(len(data)):
                print("user {} : on data point no. {}: index of table {}".format(userID, v, data[v]))
        print("===Number of data points [user, model] n_kd")
        for user in range(self._n_kd.shape[1]):
            for model in range(self._n_kd.shape[0]):
                print("user {} : model {}: number of data points {}".format(user, model, self._n_kd[model, user]))
        print("===Number of data points [model] self.param")
        for model in range(self._K):
            print("model {}: num of data points: {}".format(model, self.params[model]._X.shape))
            print(self.params[model]._Y)
        print("===Number of tables [model] m_k")
        for model in range(self._m_k.shape[0]):
            print("model {}: number of tables {}".format(model, self._m_k[model]))

    """
    sample the data to train the parameters
    @param iteration: the maximum number of gibbs sampling iteration
    """
    def sample(self, iteration):
        # sample the total data
        iter = 0
        max_iter_alpha = 40
        max_iter_gamma = 20
        # store log-likelihood for each iteration
        self.log_likelihoods = np.zeros(iteration)
        while iter < iteration:
            iter += 1

            # sample new w_alpha and s_alpha
            for iter_alpha in range(max_iter_alpha):
                self._w_alpha = np.array([np.random.beta(self._alpha+1, np.sum(self._n_dt[j])) for j in range(self._D)])
                self._s_alpha = np.array(
                    [np.random.binomial(1, np.sum(self._n_dt[j]) / (np.sum(self._n_dt[j]) + self._alpha)) for j in
                     range(self._D)])
                # sample alpha from the gamma distribution
                rate = 1./self._alpha_b - np.sum(np.log(self._w_alpha))
                self._alpha = np.random.gamma(self._alpha_a + np.sum(self._m_k) - np.sum(self._s_alpha), 1./rate)
                # print "alpha is ", self._alpha
            # sample gamma from a gamma distribution
            for iter_gamma in range(max_iter_gamma):
                self._w_gamma = np.random.beta(self._gamma+1, np.sum(self._m_k))
                pi = self._gamma_a + self._K - 1
                self._s_gamma = np.random.binomial(1, (self._gamma_b - np.log(self._w_gamma)) * np.sum(self._m_k) / (
                pi + (self._gamma_b - np.log(self._w_gamma)) * np.sum(self._m_k)))
                rate = 1./self._gamma_b - np.log(self._w_gamma)
                self._gamma = np.random.gamma(self._gamma_a + self._K - self._s_gamma, 1./rate)
                # print "gamma is ", self._gamma

            for document_index in np.random.permutation(range(self._D)):
                # sample customer assignment, see which table it should belong to
                for word_index in np.random.permutation(range(len(self._corpus_x[document_index]))):
                    self.update_params(document_index, word_index, -1)

                    # get the data at the index position
                    x = self._corpus_x[document_index][word_index]
                    y = self._corpus_y[document_index][word_index]

                    # compute the log-likelihood
                    f = np.zeros(self._K, dtype=np.float64)
                    flog = np.zeros(self._K, dtype=np.float64)
                    f_new_table = 0.
                    for k in range(self._K):
                        flog[k] = self.params[k].predictive_logpdf(x, y)
                        try:
                            assert not np.isnan(flog[k])
                        except:
                            type, value, tb = sys.exc_info()
                            traceback.print_exc()
                            pdb.post_mortem(tb)
                        f[k] = np.exp(flog[k])
                        f_new_table += self._m_k[k]*f[k]
                    base_distribution = Gaussian(X=np.zeros((0, len(x))), Y=np.zeros(0))
                    f_new_topic = np.exp(base_distribution.predictive_logpdf(x, y))
                    f_new_table += self._gamma*f_new_topic
                    f_new_table /= (np.sum(self._m_k) + self._gamma)    # normalization
                    # assert f_new_table > 0.

                    # compute the prior probability of this word sitting at every table
                    # table_probability = np.zeros(len(self._k_dt[document_index]) + 1)
                    table_probability_log = np.zeros(len(self._k_dt[document_index]) + 1, dtype=np.float64)
                    for t in range(len(self._k_dt[document_index])):
                        if self._n_dt[document_index][t] > 0:
                            # if there are some words sitting on this table,
                            # the probability will be proportional to the population
                            assigned_topic = int(self._k_dt[document_index][t])
                            assert(assigned_topic >= 0 or assigned_topic < self._K)
                            # table_probability[t] = f[assigned_topic] * self._n_dt[document_index][t]
                            table_probability_log[t] = flog[assigned_topic] + np.log(self._n_dt[document_index][t])
                        else:
                            # if there are no words sitting on this table
                            # note that it is an old table, hence the prior probability is 0, not self._alpha
                            # table_probability[t] = 0.
                            table_probability_log[t] = np.log(0.)
                    # compute the prob of current word sitting on a new table, the prior probability is self._alpha
                    # table_probability[len(self._k_dt[document_index])] = self._alpha * f_new_table
                    table_probability_log[len(self._k_dt[document_index])] = np.log(self._alpha) + np.log(f_new_table)

                    # sample a new table this word should sit in
                    table_probability = np.exp(table_probability_log)
                    assert np.sum(table_probability) > 0.

                    table_probability /= np.sum(table_probability)
                    cdf = np.cumsum(table_probability)
                    new_table = np.uint8(np.nonzero(cdf >= np.random.random())[0][0])

                    # assign current word to new table
                    self._t_dv[document_index][word_index] = new_table

                    # if current word sits on a new table, we need to get the topic of that table
                    if new_table == len(self._k_dt[document_index]):
                        # expand the vectors to fit in new table
                        self._n_dt[document_index] = np.hstack((self._n_dt[document_index], np.zeros(1)))
                        self._k_dt[document_index] = np.hstack((self._k_dt[document_index], np.zeros(1)))

                        assert(len(self._n_dt) == self._D and np.all(self._n_dt[document_index] >= 0))
                        assert(len(self._k_dt) == self._D and np.all(self._k_dt[document_index] >= 0))
                        assert(len(self._n_dt[document_index]) == len(self._k_dt[document_index]))

                        # compute the probability of this table having every topic
                        # topic_probability = np.zeros(self._K + 1)
                        topic_probability_log = np.zeros(self._K + 1, dtype=np.float64)
                        for k in range(self._K):
                            # topic_probability[k] = self._m_k[k] * f[k]
                            topic_probability_log[k] = np.log(self._m_k[k]) + flog[k]
                        # topic_probability[self._K] = self._gamma * f_new_topic
                        topic_probability_log[self._K] = np.log(self._gamma) + np.log(f_new_topic)

                        # sample a new topic this table should be assigned
                        topic_probability = np.exp(topic_probability_log)
                        assert np.sum(topic_probability) > 0.
                        topic_probability /= np.sum(topic_probability)
                        cdf = np.cumsum(topic_probability)
                        new_topic = np.uint8(np.nonzero(cdf >= np.random.random())[0][0])

                        self._k_dt[document_index][new_table] = new_topic

                        # if current table requires a new topic
                        if new_topic == self._K:
                            # expand the matrices to fit in new topic
                            self._K += 1
                            self._n_kd = np.vstack((self._n_kd, np.zeros((1, self._D)))).astype(int)
                            assert(self._n_kd.shape == (self._K, self._D))
                            self._k_dt[document_index][-1] = new_topic
                            self._m_k = np.hstack((self._m_k, np.zeros(1)))
                            assert(len(self._m_k) == self._K)
                            self.params[new_topic] = Gaussian(X=np.zeros((0, len(x))), Y=np.zeros(0))

                    self.update_params(document_index, word_index, +1)

                # sample table assignment, see which topic it should belong to
                for table_index in np.random.permutation(range(len(self._k_dt[document_index]))):
                    # if this table is not empty, sample the topic assignment of this table
                    if self._n_dt[document_index][table_index] > 0:
                        old_topic = int(self._k_dt[document_index][table_index])

                        # find the index of the words sitting on the current table
                        selected_word_index = np.nonzero(self._t_dv[document_index] == table_index)[0]
                        # find all the data associated with current table
                        selected_word = np.array([self._corpus_x[document_index][term]
                                                  for term in list(selected_word_index)])
                        selected_word_y = np.array([self._corpus_y[document_index][term]
                                                    for term in list(selected_word_index)])
                        # remove all the data in this table from their cluster
                        for x, y in zip(selected_word, selected_word_y):
                            self.params[old_topic].rm_point(x, y)

                        # compute the probability of assigning current table every topic
                        topic_probability_log = np.zeros(self._K + 1, dtype=np.float64)
                        # first compute the log-likelihood of a new topic
                        topic_probability_log[self._K] = 0.
                        for x, y in zip(selected_word, selected_word_y):
                            base_distribution = Gaussian(X=np.zeros((0, len(x))), Y=np.zeros(0))
                            topic_probability_log[self._K] += base_distribution.predictive_logpdf(x, y)
                        topic_probability_log[self._K] += np.log(self._gamma)

                        # compute the likelihood of each existing topic
                        for topic_index in range(self._K):
                            if topic_index == old_topic:
                                if self._m_k[topic_index] <= 1:
                                    # if current table is the only table assigned to current topic,
                                    # it means this topic is probably less useful or less generalizable to other documents,
                                    # it makes more sense to collapse this topic and hence assign this table to other topic.
                                    topic_probability_log[topic_index] = -1e500
                                else:
                                    # if there are other tables assigned to current topic
                                    # topic_probability[topic_index] = 0.
                                    for x, y in zip(selected_word, selected_word_y):
                                        assert self.params[topic_index].predictive_logpdf(x, y) > -np.inf
                                        topic_probability_log[topic_index] += self.params[topic_index].predictive_logpdf(x, y)
                                    # compute the prior if we move this table from this topic
                                    assert self._m_k[topic_index] - 1 > 0
                                    topic_probability_log[topic_index] += np.log(self._m_k[topic_index] - 1)
                            else:
                                # topic_probability[topic_index] = 0.
                                for x, y in zip(selected_word, selected_word_y):
                                    assert self.params[topic_index].predictive_logpdf(x, y) > -np.inf
                                    topic_probability_log[topic_index] += self.params[topic_index].predictive_logpdf(x, y)
                                # assert self._m_k[topic_index] > 0
                                topic_probability_log[topic_index] += np.log(self._m_k[topic_index])

                        # normalize the distribution and sample new topic assignment for this topic
                        # if len(np.where(topic_probability <= -np.inf)[0]) != 0:
                        #    print np.where(topic_probability <= -np.inf)[0]
                        #    print 'number of topics ', self._K
                        # assert np.all(topic_probability > -np.inf)
                        topic_probability = np.exp(topic_probability_log)
                        assert np.sum(topic_probability) > 0.
                        topic_probability /= np.sum(topic_probability)

                        cdf = np.cumsum(topic_probability)
                        rdm = np.random.random()
                        if len(np.nonzero(cdf >= rdm)[0]) == 0:
                            print(topic_probability)
                        new_topic = np.uint8(np.nonzero(cdf >= rdm)[0][0])

                        # if the table is assigned to a new topic
                        if new_topic != old_topic:
                            # assign this table to new topic
                            self._k_dt[document_index][table_index] = new_topic

                            # if this table starts a new topic, expand all matrix
                            if new_topic == self._K:
                                self._K += 1
                                self._n_kd = np.vstack((self._n_kd, np.zeros((1, self._D)))).astype(int)
                                assert(self._n_kd.shape == (self._K, self._D))
                                self._m_k = np.hstack((self._m_k, np.zeros(1)))
                                assert(len(self._m_k) == self._K)
                                self.params[new_topic] = Gaussian(X=np.zeros((0, len(selected_word[0]))), Y=np.zeros(0))

                            # adjust the statistics of all model parameter
                            self._m_k[old_topic] -= 1
                            self._m_k[new_topic] += 1
                            self._n_kd[old_topic, document_index] -= self._n_dt[document_index][table_index]
                            self._n_kd[new_topic, document_index] += self._n_dt[document_index][table_index]
                        # add data point to the cluster
                        for x, y in zip(selected_word, selected_word_y):
                            self.params[new_topic].add_point(x, y)

            # compact all the parameters, including removing unused topics and unused tables
            self.compact_params()

            if self._flag_compute_loglik:
                # print "gamma is %.2f, alpha is %.2f" % (self._gamma, self._alpha)
                self.log_likelihoods[iter-1] = self.get_logpdf()
                '''
                if iter >= 2:
                    if self.log_likelihoods[iter-1] < self.log_likelihoods[iter-2]:
                        print "warning: log-likelihood is decreasing..."
                '''
            if iter > 0 and iter % self._snapshot_interval == 0:
                print("sampling in progress %2d%%" % (100 * iter / iteration))
                print("total number of topics %i " % self._K)
                if self._flag_compute_loglik:
                    print("gamma is %.2f, alpha is %.2f" % (self._gamma, self._alpha))
                    self.log_likelihoods[iter - 1] = self.get_logpdf()
                    print('model log-likelihood is ', self.log_likelihoods[iter-1])

    """
    @param document_index: the document index to update
    @param word_index: the word index to update
    @param update: the update amount for this document and this word
    @attention: the update table index and topic index is retrieved from self._t_dv and self._k_dt, so make sure these values were set properly before invoking this function
    """
    def update_params(self, document_index, word_index, update):
        # retrieve the table_id of the current word of current document
        table_id = self._t_dv[document_index][word_index]
        # retrieve the topic_id of the table that current word of current document sit on
        topic_id = int(self._k_dt[document_index][table_id])
        # get the data at the word_index of the document_index
        x = self._corpus_x[document_index][word_index]
        y = self._corpus_y[document_index][word_index]
        self._n_dt[document_index][table_id] += update
        assert(np.all(self._n_dt[document_index] >= 0))
        if update == -1:
            self.params[topic_id].rm_point(x, y)
        elif update == 1:
            self.params[topic_id].add_point(x, y)
        self._n_kd[topic_id, document_index] += update
        assert(np.all(self._n_kd >= 0))

        # if current table in current document becomes empty
        if update == -1 and self._n_dt[document_index][table_id] == 0:
            # adjust the table counts
            self._m_k[topic_id] -= 1

        # if a new table is created in current document
        if update == 1 and self._n_dt[document_index][table_id] == 1:
            # adjust the table counts
            self._m_k[topic_id] += 1

        assert(np.all(self._m_k >= 0))
        assert(np.all(self._k_dt[document_index] >= 0))

    def compact_params(self):
        # find unused and used topics
        unused_topics = np.nonzero(self._m_k == 0)[0]
        used_topics = np.nonzero(self._m_k != 0)[0]

        self._K -= len(unused_topics)
        assert(self._K >= 1 and self._K == len(used_topics))

        self._n_kd = np.delete(self._n_kd, unused_topics, axis=0)
        assert(self._n_kd.shape == (self._K, self._D))

        self._m_k = np.delete(self._m_k, unused_topics)
        assert(len(self._m_k) == self._K)

        for k in range(len(used_topics)):
            self.params[k] = self.params.pop(used_topics[k])
        for key in range(len(self.params)):
            if key >= self._K and key in unused_topics:
                del self.params[key]
        for d in range(self._D):
            # find the unused and used tables
            unused_tables = np.nonzero(self._n_dt[d] == 0)[0]
            used_tables = np.nonzero(self._n_dt[d] != 0)[0]

            self._n_dt[d] = np.delete(self._n_dt[d], unused_tables)
            self._k_dt[d] = np.delete(self._k_dt[d], unused_tables)

            # shift down all the table indices of all words in current document
            # @attention: shift the used tables in ascending order only.
            for t in range(len(self._n_dt[d])):
                self._t_dv[d][np.nonzero(self._t_dv[d] == used_tables[t])[0]] = t

            # shrink down all the topics indices of all tables in current document
            # @attention: shrink the used topics in ascending order only.
            for k in range(self._K):
                self._k_dt[d][np.nonzero(self._k_dt[d] == used_topics[k])[0]] = k

    # def get_logpdf(self, data=None):
    #     if data is None:
    #         data = self._corpus
    #     weights, dists = dict2mix(self.params)
    #     tmp = [all_loglike(X, weights, dists) for X in data]
    #     loglik = np.sum(tmp)
    #     # add likelihood of alpha and gamma
    #     loglik += (self._alpha_a - 1)*np.log(self._alpha) - self._alpha/self._alpha_b - self._alpha_a*np.log(self._alpha_b) - ssp.gammaln(self._alpha_a)
    #     loglik += (self._gamma_a - 1)*np.log(self._gamma) - self._gamma/self._gamma_b - self._gamma_a*np.log(self._gamma_b) - ssp.gammaln(self._gamma_a)
    #     return loglik

    def pickle(self, path, filename):
        with file(path + filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def unpickle(self, path):
        with file(path, 'rb') as f:
            return pickle.load(f)