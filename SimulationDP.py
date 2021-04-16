import numpy as np
import json
from random import sample, shuffle
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager
# Import Bandit algorithms
from lib.LinUCB import LinUCB
from lib.AdaptiveThompson import AdaptiveThompson
from lib.dLinUCB import dLinUCB
from lib.CLUB import CLUBAlgorithm
from lib.oracleLinUCB import oracleLinUCB
from lib.CoDBand import CoDBand
from lib.SCLUB import SCLUB

class simulateOnlineData(object):
    def __init__(self, namelabel, context_dimension, testing_iterations, maximum_change_schedule,
                 minimum_change_schedule,
                 articles,
                 users,
                 global_parameter_set,
                 parameter_index_for_users,
                 alpha_0,
                 batchSize=1000,
                 noise=lambda: 0,
                 type_='UniformTheta',
                 signature='',
                 poolArticleSize=10,
                 NoiseScale=0.0,
                 Plot=False,
                 Write_to_File=False):
        self.namelabel = namelabel
        self.simulation_signature = signature
        self.type = type_
        self.context_dimension = context_dimension
        self.testing_iterations = testing_iterations
        self.maximum_change_schedule = maximum_change_schedule
        self.minimum_change_schedule = minimum_change_schedule
        self.noise = noise
        self.NoiseScale = NoiseScale
        self.articles = articles
        self.users = users
        self.global_parameter_set = global_parameter_set
        self.parameter_index_for_users = parameter_index_for_users
        self.global_parameter_popularity = [0]*self.global_parameter_set.shape[0]
        for index in self.parameter_index_for_users:
            self.global_parameter_popularity[index] += 1
        # self.global_parameter_popularity = np.array(self.global_parameter_popularity)
        self.alpha_0 = alpha_0
        self.dimension = self.global_parameter_set.shape[1]
        self.poolArticleSize = poolArticleSize
        self.batchSize = batchSize

        self.Plot = Plot
        self.Write_to_File = Write_to_File

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, "Pool", len(self.articlePool), " Elapsed time",
              datetime.datetime.now() - self.startTime)

    def regulateArticlePool(self):
        # Randomly generate articles
        self.articlePool = sample(self.articles, self.poolArticleSize)

    def getReward(self, user, pickedArticle):
        return np.dot(user.theta, pickedArticle.featureVector)

    def GetOptimalReward(self, user, articlePool):
        maxReward = float('-inf')
        for x in articlePool:
            reward = self.getReward(user, x)
            if reward > maxReward:
                maxReward = reward
        return maxReward

    def getL2Diff(self, x, y):
        return np.linalg.norm(x - y)  # L2 norm

    def runAlgorithms(self, algorithms, startTime):
        self.startTime = startTime
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')
        filenameWriteRegret = os.path.join(save_address, self.namelabel + "_" + 'AccRegret' + timeRun + '.csv')
        filenameWritePara = os.path.join(save_address, self.namelabel + "_" + 'ParameterEstimation' + timeRun + '.csv')

        tim_ = []
        BatchCumlateRegret = {}
        AlgRegret = {}
        ThetaDiffList = {}
        ThetaDiff = {}

        # Initialization
        userSize = len(self.users)
        for alg_name, alg in algorithms.items():
            AlgRegret[alg_name] = []
            BatchCumlateRegret[alg_name] = []
            if alg.CanEstimateUserPreference:
                ThetaDiffList[alg_name] = []

        if self.Write_to_File:
            with open(filenameWriteRegret, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(filenameWritePara, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) + 'Theta' for alg_name in ThetaDiffList.keys()]))
                f.write('\n')

        # Shuffle the candidate arm pool
        shuffle(self.articles)

        users_change_schedule = {}  # change schedule for each user
        users_change_time = {}  # time of change for each user
        ThetaList = {}

        for u in self.users:
            users_change_time[u.id] = [0]
            ThetaList[u.id] = [u.theta]

            # initialize change schedule for each user
            change_schedule = np.random.randint(self.minimum_change_schedule, self.maximum_change_schedule + 1)
            users_change_schedule[u.id] = change_schedule

        # print("Initial global parameter popularity:")
        # print(self.global_parameter_popularity)

        for iter_ in range(self.testing_iterations):
            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiff[alg_name] = 0

            iter_precision = {}
            iter_recall = {}
            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserCluster:
                    iter_precision[alg_name] = []
                    iter_recall[alg_name] = []

            for u in self.users:
                # Simulate asynchronous changes for each user
                if iter_ > (users_change_time[u.id][-1] + users_change_schedule[u.id]):
                    users_change_time[u.id].append(iter_)
                    # sample a new change schedule for this user to change next time
                    users_change_schedule[u.id] = np.random.randint(self.minimum_change_schedule,
                                                                    self.maximum_change_schedule + 1)
                    # sample a new parameter that is different from current one for this user from the global parameter set
                    # self.global_parameter_popularity[self.parameter_index_for_users[u.id]] -= 1
                    prob = self.global_parameter_popularity + [self.alpha_0]
                    # print(self.global_parameter_popularity)
                    # print(prob)
                    prob = prob / np.sum(prob)

                    new_parameter_index = np.random.choice(len(self.global_parameter_popularity) + 1, size=None, replace=True, p=prob)
                    assert new_parameter_index != len(self.global_parameter_popularity) + 1
                    if new_parameter_index == len(self.global_parameter_popularity):
                        # print("new parameter index!")
                        # generate new theta
                        thetaVector = gaussianFeature(self.dimension, argv={'l2_limit': 1})
                        l2_norm = np.linalg.norm(thetaVector, ord=2)
                        new_theta = thetaVector / l2_norm

                        dist_to_all_existing_big = all([np.linalg.norm(new_theta - existing_theta) >= 0.9 for existing_theta in self.global_parameter_set])
                        while (not dist_to_all_existing_big):
                            thetaVector = gaussianFeature(self.dimension, argv={'l2_limit': 1})
                            l2_norm = np.linalg.norm(thetaVector, ord=2)
                            new_theta = thetaVector / l2_norm
                            dist_to_all_existing_big = all(
                                [np.linalg.norm(new_theta - existing_theta) >= 0.9 for existing_theta in
                                global_parameter_set])
                        # print(self.global_parameter_set.shape)
                        # print(new_theta.shape)
                        # print(new_theta.reshape([1,-1]).shape)
                        self.global_parameter_set = np.concatenate((self.global_parameter_set, new_theta.reshape((1,-1))), axis=0)
                        self.global_parameter_popularity.append(1)
                        assert self.global_parameter_set.shape[0] == len(self.global_parameter_popularity)
                        self.parameter_index_for_users[u.id] = new_parameter_index
                        u.theta = self.global_parameter_set[new_parameter_index]
                        # print("number of parameters now: {}".format(self.global_parameter_set.shape[0]))
                    else:
                        # print("old parameter index!")
                        self.global_parameter_popularity[new_parameter_index] += 1
                        self.parameter_index_for_users[u.id] = new_parameter_index
                        u.theta = self.global_parameter_set[new_parameter_index]
                    print(self.global_parameter_popularity)
                self.regulateArticlePool()  # select random articles
                noise = self.noise()
                OptimalReward = self.GetOptimalReward(u, self.articlePool)
                OptimalReward += noise

                # Get true cluster of u
                true_cluster = []
                for u_j in self.users:
                    if self.parameter_index_for_users[u.id] == self.parameter_index_for_users[u_j.id]:
                        assert np.linalg.norm(u_j.theta - u.theta) <= 0.01
                        true_cluster.append(u_j.id)

                for alg_name, alg in algorithms.items():
                    # Observe the candiate arm pool and algoirhtm makes a decision
                    if alg_name == "oracleLinUCB":
                        pickedArticle = alg.decide(self.articlePool, u.id, self.parameter_index_for_users[u.id])
                    else:
                        pickedArticle = alg.decide(self.articlePool, u.id)

                    if alg_name == "CoDBand" or alg_name == "SCLUB":
                        if alg_name == "CoDBand":
                            estimated_cluster = [u.userID for u in alg.cluster]
                        if alg_name == "SCLUB":
                            estimated_cluster = alg.cluster
                            # print(estimated_cluster)
                        # compute precision and recall for cluster identification
                        # which users have the same parameter as current user
                        TP_count = 0
                        for e_neighbor in estimated_cluster:
                            if e_neighbor in true_cluster:
                                TP_count += 1
                        TP_FP_count = len(estimated_cluster)
                        TP_FN_count = len(true_cluster)
                        precision = TP_count / TP_FP_count
                        recall = TP_count / TP_FN_count
                        iter_precision[alg_name].append(precision)
                        iter_recall[alg_name].append(recall)

                    # Get the feedback from the environment
                    reward = self.getReward(u, pickedArticle) + noise
                    # The feedback/observation will be fed to the algorithm to further update the algorithm's model estimation
                    if alg_name == 'CLUB':
                        alg.updateParameters(pickedArticle.featureVector, reward, u.id)
                        n_components = alg.updateGraphClusters(u.id, 'False')
                    elif alg_name == "oracleLinUCB":
                        alg.updateParameters(pickedArticle, reward, u.id, self.parameter_index_for_users[u.id])
                    else:
                        alg.updateParameters(pickedArticle, reward, u.id)
                    # Calculate and record the regret
                    regret = OptimalReward - reward
                    AlgRegret[alg_name].append(regret)

                    # Update parameter estimation record
                    if alg.CanEstimateUserPreference:
                        ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))

            for alg_name, alg in algorithms.items():
                if alg.CanEstimateUserPreference:
                    ThetaDiffList[alg_name] += [ThetaDiff[alg_name] / userSize]

            if iter_ % self.batchSize == 0:
                self.batchRecord(iter_)
                tim_.append(iter_)

                # Print out averaged precision and recall over users for this iteration
                for alg_name, alg in algorithms.items():
                    if alg_name == 'CoDBand':
                        print("alg_name: {}, cluster identification precision: {:.2}, recall: {:.2}".format(alg_name,
                                                                                                            np.mean(
                                                                                                                iter_precision[
                                                                                                                    alg_name]),
                                                                                                            np.mean(
                                                                                                                iter_recall[
                                                                                                                    alg_name])))

                for alg_name in algorithms.keys():
                    BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))

                if self.Write_to_File:
                    with open(filenameWriteRegret, 'a+') as f:
                        f.write(str(iter_))
                        f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                        f.write('\n')
                    with open(filenameWritePara, 'a+') as f:
                        f.write(str(iter_))
                        f.write(',' + ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.keys()]))
                        f.write('\n')

        # print actual and detected change points for each user
        for alg_name in algorithms.keys():
            if alg_name == 'adTS':
                print("============= adTS detected change =============")
                for u in self.users:
                    print("User {} actual change points:   {}".format(u.id, str(users_change_time[u.id])))
                    print("User {} detected change points: {}".format(u.id, algorithms[alg_name].users[u.id].changes))
            if alg_name == 'dLinUCB':
                print("============= dLinUCB detected change =============")
                for u in self.users:
                    print("User {} actual change points:   {}".format(u.id, str(users_change_time[u.id])))
                    print("User {} detected change points: {}".format(u.id, algorithms[alg_name].users[u.id].newUCBs))
            if alg_name == 'CoDBand':
                print("============= CoDBand detected change =============".format(alg_name))
                for u in self.users:
                    print("User {} actual change points:   {}".format(u.id, str(users_change_time[u.id])))
                    print("User {} detected change points: {}".format(u.id, algorithms[alg_name].userModels[u.id].detectedChangePoints))
        if self.Plot:  # only plot
            linestyles = ['o-', 's-', '*-', '>-', '<-', 'g-', '.-', 'o-', 's-', '*-']
            markerlist = ['.', ',', 'o', 's', '*', 'v', '>', '<']
            fig, ax = plt.subplots()
            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                ax.plot(tim_, BatchCumlateRegret[alg_name], linewidth=2, marker=markerlist[count], markevery=400,
                        label=labelName)
                count += 1
            ax.legend(loc='upper right')
            ax.set(xlabel='Iteration', ylabel='Regret',
                   title='Regret over iterations')
            ax.grid()
            plt.savefig(os.path.join(save_address, self.namelabel + str(timeRun) + '.png'))
            plt.show()
        print("Accumulated regret:")
        for alg_name in algorithms.keys():
            print('%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1]))
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--namelabel', dest='namelabel', help='Name')
    parser.add_argument('--T', dest='T', help='total number of iterations')
    parser.add_argument('--SMIN', dest='SMIN', help='SMIN')
    parser.add_argument('--SMAX', dest='SMAX', help='SMAX')
    parser.add_argument('--sigma', dest='sigma', help='std of gaussian noise in reward')
    parser.add_argument('--n', dest='n', help='number of users')
    parser.add_argument('--m', dest='m', help='number of unique parameters')

    args = parser.parse_args()

    config = {}
    if args.T:
        config["testing_iterations"] = int(args.T)
    else:
        config["testing_iterations"] = 3000
    if args.SMIN:
        config["minimum_change_schedule"] = int(args.SMIN)
    else:
        config["minimum_change_schedule"] = 500
    if args.SMAX:
        config["maximum_change_schedule"] = int(args.SMAX)
    else:
        config["maximum_change_schedule"] = 3000
    if args.sigma:
        config["NoiseScale"] = float(args.sigma)
    else:
        config["NoiseScale"] = 0.1  # standard deviation of Gaussian noise in reward
    if args.n:
        config["n_users"] = int(args.n)
    else:
        config["n_users"] = 10  # number of users
    if args.m:
        config["UserGroups"] = int(args.m)
    else:
        config["UserGroups"] = 2  # number of unique parameters
    if args.namelabel:
        namelabel = str(args.namelabel)
    else:
        namelabel = "DP"+"n" + str(config["n_users"]) + "_" + "m" + str(config["UserGroups"]) + "_" + "SMIN" + str(
            config["minimum_change_schedule"]) + "_" + "SMAX" + str(
            config["maximum_change_schedule"]) + "_" + "Sigma" + str(config["NoiseScale"])

    # Some other environment settings
    config["context_dimension"] = 25  # feature dimension
    config["n_articles"] = 1000  # Total number of arms/articles
    config["ArticleGroups"] = 0
    config["gamma"] = 0.9  # gap between unique parameters
    config["poolSize"] = 10  # number of arms in the armpool in each itereation
    config["alpha_0"] = 4
    # Output
    batchSize = 1  # The batchsize when calculating and plotting the regret
    Write_to_File = True
    Plot = True

    # Algorithm parameters
    config["lambda_"] = 0.1  # regularization in ridge regression
    # CLUB
    config["CLUB_alpha"] = 0.1
    config["CLUB_alpha_2"] = 2.0
    config["cluster_init"] = "Complete"  # or "Erdos-Renyi"
    # AdTS
    config["AdTS_Window"] = 200
    config["v"] = 1
    # LinUCB
    config["alpha"] = 0.3
    # dLinUCB
    config["tau"] = 20  # size of sliding window
    config["delta_1"] = 1e-1
    config["delta_2"] = 1e-1
    config["dLinUCB_alpha"] = 0.6
    config["tilde_delta_1"] = config["delta_1"] #/ 5.0  # tilde_delta_1 should be a number between 0 and self.delta_1

    #CoDBand
    config["memory_size"] = 60

    # Generate user and item vectors
    userFilename = os.path.join(sim_files_folder, "users_" + str(config["n_users"]) + "context_" + str(
        config["context_dimension"]) + "Ugroups" + str(config["UserGroups"]) + ".json")
    UM = UserManager(config["context_dimension"], config["n_users"], config["gamma"], UserGroups=config["UserGroups"],
                     thetaFunc=gaussianFeature, argv={'l2_limit': 1})
    users, global_parameter_set, parameter_index_for_users = UM.simulateThetafromUsers()

    articlesFilename = os.path.join(sim_files_folder, "articles_" + str(config["n_articles"]) + "context_" + str(
        config["context_dimension"]) + "Agroups" + str(config["ArticleGroups"]) + ".json")
    AM = ArticleManager(config["context_dimension"], n_articles=config["n_articles"],
                        ArticleGroups=config["ArticleGroups"],
                        FeatureFunc=gaussianFeature, argv={'l2_limit': 1})
    articles = AM.simulateArticlePool()

    for i in range(len(articles)):
        articles[i].contextFeatureVector = articles[i].featureVector[:config["context_dimension"]]

    simExperiment = simulateOnlineData(namelabel=namelabel,
                                       context_dimension=config["context_dimension"],
                                       testing_iterations=config["testing_iterations"],
                                       maximum_change_schedule=config["maximum_change_schedule"],
                                       minimum_change_schedule=config["minimum_change_schedule"],
                                       articles=articles,
                                       users=users,
                                       global_parameter_set=global_parameter_set,
                                       parameter_index_for_users=parameter_index_for_users,
                                       alpha_0 = config["alpha_0"],
                                       noise=lambda: np.random.normal(scale=config["NoiseScale"]),
                                       batchSize=batchSize,
                                       type_="UniformTheta",
                                       signature=AM.signature,
                                       poolArticleSize=config["poolSize"],
                                       NoiseScale=config["NoiseScale"],
                                       Plot=Plot,
                                       Write_to_File=Write_to_File)

    print("Starting for ", simExperiment.simulation_signature)

    algorithms = {}
    algorithms['oracleLinUCB'] = oracleLinUCB(dimension=config["context_dimension"], alpha=config["alpha"],
                                              lambda_=config["lambda_"], NoiseScale=config["NoiseScale"],
                                              delta_1=config["delta_1"])
    algorithms['LinUCB'] = LinUCB(dimension=config["context_dimension"], alpha=config["alpha"],
                                  lambda_=config["lambda_"], NoiseScale=config["NoiseScale"])
    algorithms['adTS'] = AdaptiveThompson(dimension=config["context_dimension"], AdTS_Window=config["AdTS_Window"],
                                          AdTS_CheckInter=50, v=config["v"])
    algorithms['dLinUCB'] = dLinUCB(dimension=config["context_dimension"], alpha=config["dLinUCB_alpha"],
                                    lambda_=config["lambda_"], NoiseScale=config["NoiseScale"], tau=config["tau"],
                                    delta_1=config["delta_1"], delta_2=config["delta_2"],
                                    tilde_delta_1=config["tilde_delta_1"])
    algorithms['CLUB'] = CLUBAlgorithm(dimension=config["context_dimension"], alpha=config["CLUB_alpha"],
                                       lambda_=config["lambda_"], n=config["n_users"],
                                       alpha_2=config["CLUB_alpha_2"], cluster_init=config["cluster_init"])
    algorithms['SCLUB'] = SCLUB(nu=config["n_users"], d=config["context_dimension"], NoiseScale=config["NoiseScale"], lambda_=config["lambda_"])
    algorithms['CoDBand'] = CoDBand(v=1, d=config["context_dimension"], lambda_=config["lambda_"], NoiseScale=config["NoiseScale"], alpha_prior={'a': 7.5, 'b': 1.}, tau_cd=config["tau"], alpha=config["alpha"], memory_size=config["memory_size"])

    startTime = datetime.datetime.now()
    with open(os.path.join(save_address, 'Config' + startTime.strftime('_%m_%d_%H_%M_%S') + '.json'), 'w') as fp:
        json.dump(config, fp)
    simExperiment.runAlgorithms(algorithms, startTime)
