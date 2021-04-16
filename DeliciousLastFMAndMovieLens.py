import copy
import numpy as np
import random
from random import sample, shuffle, choice
import datetime
import os.path
import json
import matplotlib.pyplot as plt
import argparse

# local address to different datasets
from conf import *
from operator import truediv
#Stationary Bandit algorithms
from lib.LinUCB import LinUCB, UniformLinUCB
from lib.AdaptiveThompson import AdaptiveThompson
from lib.dLinUCB import dLinUCB
from lib.CLUB import CLUBAlgorithm
from lib.SCLUB import SCLUB
from lib.CoDBand import CoDBand

# real dataset
from dataset_utils.LastFM_util_functions_2 import readFeatureVectorFile, parseLine
from scipy.special import erfinv

class Article():	
    def __init__(self, aid, FV=None):
        self.article_id = aid
        self.contextFeatureVector = FV
        self.featureVector = FV

class experimentOneRealData(object):
    def __init__(self, namelabel, dataset, context_dimension, batchSize = 1 ,plot = True, Write_to_File = False):
        
        self.namelabel = namelabel
        assert dataset in ["LastFM", "Delicious", "MovieLens"]
        self.dataset = dataset
        self.context_dimension = context_dimension
        self.Plot = plot
        self.Write_to_File = Write_to_File
        self.batchSize = batchSize
        if self.dataset == 'LastFM':
            self.relationFileName = LastFM_relationFileName
            self.address = LastFM_address
            self.save_address = LastFM_save_address
            FeatureVectorsFileName = LastFM_FeatureVectorsFileName
            self.event_fileName = self.address + "/simulatedNonstationaryClusterUsers.dat"
        elif self.dataset == 'Delicious':
            # self.relationFileName = Delicious_relationFileName
            self.address = Delicious_address
            self.save_address = Delicious_save_address
            FeatureVectorsFileName = Delicious_FeatureVectorsFileName
            self.event_fileName = self.address + "/simulatedNonstationaryClusterUsers_N20Gamma59_ObsMoreThan50.dat"
        else:
            self.relationFileName = MovieLens_relationFileName
            self.address = MovieLens_address
            self.save_address = MovieLens_save_address
            FeatureVectorsFileName = MovieLens_FeatureVectorsFileName
            self.event_fileName = self.address + "/randUserOrderedTime.dat"
        # Read Feature Vectors from File
        self.FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
        self.articlePool = []
    
    def getL2Diff(self, x, y):
        return np.linalg.norm(x-y) # L2 norm

    def batchRecord(self, iter_):
        print("Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime)

    def runAlgorithms(self, algorithms, startTime):
        self.startTime = startTime
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')

        filenameWriteReward = os.path.join(self.save_address, 'AccReward' + str(self.namelabel) + timeRun + '.csv')
        end_num = 0
        while os.path.exists(filenameWriteReward):
            filenameWriteReward = os.path.join(self.save_address, 'AccReward' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1
        tim_ = []
        AlgReward = {}
        BatchCumlateReward = {}
        AlgReward["random"] = []
        BatchCumlateReward["random"] = []
        for alg_name, alg in algorithms.items():
            AlgReward[alg_name] = []
            BatchCumlateReward[alg_name] = []

        if self.Write_to_File:
            with open(filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

        userIDSet = set()
        with open(self.event_fileName, 'r') as f:
            f.readline()
            iter_ = 0
            for _, line in enumerate(f, 1):
                userID, _, pool_articles = parseLine(line)
                if userID not in userIDSet:
                    userIDSet.add(userID)
                # ground truth chosen article
                article_id_chosen = int(pool_articles[0])
                # Construct arm pool
                self.article_pool = []
                for article in pool_articles:
                    article_id = int(article.strip(']'))
                    article_featureVector = self.FeatureVectors[article_id]
                    article_featureVector =np.array(article_featureVector, dtype=float)
                    assert type(article_featureVector) == np.ndarray
                    assert article_featureVector.shape == (self.context_dimension,)
                    self.article_pool.append(Article(article_id, article_featureVector))

                # Random strategy
                RandomPicked = choice(self.article_pool)
                if RandomPicked.article_id == article_id_chosen:
                    reward = 1
                else:
                    reward = 0  # avoid division by zero
                AlgReward["random"].append(reward)

                for alg_name, alg in algorithms.items():
                    #Observe the candiate arm pool and algoirhtm makes a decision
                    pickedArticle = alg.decide(self.article_pool, userID)
                    # Get the feedback by looking at whether the selected arm by alg is the same as that of ground truth
                    if pickedArticle.article_id == article_id_chosen:
                        reward = 1
                    else:
                        reward = 0
                    #The feedback/observation will be fed to the algorithm to further update the algorithm's model estimation
                    alg.updateParameters(pickedArticle, reward, userID)
                    if alg_name == 'CLUB':
                        n_components = alg.updateGraphClusters(userID, 'False')
                    # Record the reward
                    AlgReward[alg_name].append(reward)

                if iter_ % self.batchSize == 0:
                    self.batchRecord(iter_)
                    tim_.append(iter_)
                    BatchCumlateReward["random"].append(sum(AlgReward["random"]))
                    for alg_name in algorithms.keys():
                        BatchCumlateReward[alg_name].append(sum(AlgReward[alg_name]))

                        # if 'dLinUCB' in alg_name:
                        #     print("============ dLinUCB detected change ===========")
                        #     # for u in userIDSet:
                        #     print("User {} detected change points: {}".format(userID, algorithms[alg_name].users[userID].newUCBs))
                        # if 'CoDBand' in alg_name:
                        #     print("============== {} detected change ==============".format(alg_name))
                        #     # for u in userIDSet:
                        #     print("User {} detected change points: {}".format(userID, algorithms[alg_name].userModels[userID].detectedChangePoints))

                    if self.Write_to_File:
                        with open(filenameWriteReward, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(BatchCumlateReward[alg_name][-1]) for alg_name in list(algorithms.keys())+["random"]]))
                            f.write('\n')
                iter_ += 1

        cp_path = os.path.join(self.save_address, "detectedChangePoints" + str(self.namelabel) + str(timeRun) + '.txt')

        end_num = 0
        while os.path.exists(cp_path):
            cp_path = os.path.join(self.save_address, "detectedChangePoints" + str(self.namelabel) + str(timeRun) + str(end_num) + '.txt')
            end_num += 1

        with open(cp_path, "w") as text_file:

            for alg_name in algorithms.keys():
                if 'adTS' in alg_name:
                    print("============= adTS detected change =============", file=text_file)
                    for u in userIDSet:
                        print("User {} detected change points: {}".format(u, algorithms[alg_name].users[u].changes), file=text_file)
                if 'dLinUCB' in alg_name:
                    print("============ dLinUCB detected change ===========", file=text_file)
                    for u in userIDSet:
                        print("User {} detected change points: {}".format(u, algorithms[alg_name].users[u].newUCBs), file=text_file)
                if 'CoDBand' in alg_name:
                    print("============== {} detected change ==============".format(alg_name), file=text_file)
                    for u in userIDSet:
                        print("User {} detected change points: {}".format(u, algorithms[alg_name].userModels[u].detectedChangePoints), file=text_file)

        if self.Plot: # only plot
            linestyles = ['o-', 's-', '*-','>-','<-','g-', '.-', 'o-', 's-', '*-']
            markerlist = ['.', ',', 'o', 's', '*', 'v', '>', '<']

            fig, ax = plt.subplots()

            # print(len(BatchCumlateReward))

            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                ax.plot(tim_, [x/(y+1) for x, y in zip(BatchCumlateReward[alg_name], BatchCumlateReward["random"])], linewidth = 1, marker = markerlist[count], markevery = 2000,  label = labelName)
                # ax.plot(tim_, [x for x in BatchCumlateReward[alg_name]], linewidth = 2, marker = markerlist[count], markevery = 400,  label = labelName)
                count +=1
            # labelName = "random"
            # ax.plot(tim_, BatchCumlateReward["random"], linewidth = 2, marker = markerlist[count], markevery = 400,  label = labelName)
            count +=1
            ax.legend(loc = 'upper right')
            ax.set(xlabel='Iteration', ylabel='Reward',
                   title='Reward over iterations')
            ax.grid()
            plt_path = os.path.join(self.save_address, str(self.namelabel) + str(timeRun) + '.png')

            end_num = 0
            while os.path.exists(plt_path):
                plt_path = os.path.join(self.save_address, str(self.namelabel) + str(timeRun) + str(end_num) + '.png')
                end_num += 1

            plt.savefig(plt_path)
            plt.show()
        for alg_name in algorithms.keys():
            print('%s: %.2f' % (alg_name, BatchCumlateReward[alg_name][-1]))

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
    parser.add_argument('--namelabel', dest='namelabel', help='Name')
    parser.add_argument('--dataset', default='Delicious', dest='dataset', help='dataset')

    parser.add_argument('--dCLUB_alpha', dest='dCLUB_alpha', help='dCLUB_alpha')
    parser.add_argument('--dLinUCB_alpha', dest='dLinUCB_alpha', help='dLinUCB_alpha')
    parser.add_argument('--eta', dest='eta', help='eta')
    parser.add_argument('--dCLUB_NoiseScale', dest='dCLUB_NoiseScale', help='dCLUB_NoiseScale')
    parser.add_argument('--dLinUCB_NoiseScale', dest='dLinUCB_NoiseScale', help='dLinUCB_NoiseScale')
    parser.add_argument('--tau', dest='tau', help='tau')
    parser.add_argument('--scaling', dest='scaling', help='scaling')

    parser.add_argument('--detectionThreshold', dest='detectionThreshold', help='detectionThreshold')
    parser.add_argument('--clusteringThreshold', dest='clusteringThreshold', help='clusteringThreshold')
    parser.add_argument('--clusterIdentificationObsNumThreshold', dest='clusterIdentificationObsNumThreshold', help='clusterIdentificationObsNumThreshold')
    parser.add_argument('--AdTS_Window', dest='AdTS_Window', help='AdTS_Window')
    parser.add_argument('--v', dest='v', help='v')
    parser.add_argument('--CLUB_alpha_2', dest='CLUB_alpha_2', help='CLUB_alpha_2')
    parser.add_argument('--CLUB_alpha', dest='CLUB_alpha', help='CLUB_alpha')
    parser.add_argument('--cluster_init', dest='cluster_init', help='cluster_init')

    args = parser.parse_args()
    algName = str(args.alg)
    namelabel = str(args.namelabel)
    dataset = str(args.dataset)
    #Configuration about the environment
    Write_to_File = True
    plot = True
    config = {}

    # CLUB
    if args.CLUB_alpha:
        config["CLUB_alpha"] = float(args.CLUB_alpha)
    else:
        config["CLUB_alpha"] = 0.1
    if args.CLUB_alpha_2:
        config["CLUB_alpha_2"] = float(args.CLUB_alpha_2)  # Noise in the feedback
    else:
        config["CLUB_alpha_2"] = 1.0
    if args.cluster_init:
        config["cluster_init"] = str(args.cluster_init)
    else:
        config["cluster_init"] = "Complete" # or "Erdos-Renyi"

    config["n_users"] = 54

    #Parameters of LinUCB
    config["context_dimension"] = 25  # Feature dimension
    config["lambda_"] = 0.2   # regularization in ridge regression
    config["LinUCB_alpha"] = 0.3

    if args.tau:
        config["tau"] = int(args.tau)  # Noise in the feedback
    else:
        config["tau"] = 1000

    # dLinUCB related
    if args.dLinUCB_alpha:
        config["dLinUCB_alpha"] = float(args.dLinUCB_alpha)     # The coefficient for exploration in LinUCB
    else:
        config["dLinUCB_alpha"] = 0.15    # The coefficient for exploration in LinUCB  0.3
    if args.dLinUCB_NoiseScale:
        config["dLinUCB_NoiseScale"] = float(args.dLinUCB_NoiseScale)  # Noise in the feedback
    else:
        config["dLinUCB_NoiseScale"] = 0.3
    if args.eta:
        config["eta"] = float(args.eta)
    else:
        config["eta"] =  np.sqrt(2.0) * (config["dLinUCB_NoiseScale"]) * erfinv(1.0 - 1e-1) #0.3  # 0.3

    config["delta_1"] = 1e-1  # upper bound probability that reward estimation error will be smaller than CB
    config["delta_2"] = 1e-1  # upper bound probability of false alarm in change detection
    config["tilde_delta_1"] = config["delta_1"] #/ 5 #tilde_delta_1 should be a number between 0 and self.delta_1

    # adTS related
    if args.AdTS_Window:
        config["AdTS_Window"] = float(args.AdTS_Window)     # The coefficient for exploration in LinUCB
    else:
        config["AdTS_Window"] = 1000     # The coefficient for exploration in LinUCB  0.3
    if args.AdTS_Window:
        config["v"] = float(args.v)     # The coefficient for exploration in LinUCB
    else:
        config["v"] = 10     # The coefficient for exploration in ts  0.3

    # CoDBand related
    config["memory_size"] = 70
    config["alpha_prior"] = {'a': 15, 'b': 1.}
    config["CoDBand_NoiseScale"] = 0.5
    config["disable_change_detector"] = True
    config["true_alpha_0"] = None
    config["CoDBand_v"] = 0.1
    config["CoDBand_eta"] = np.sqrt(2.0) * (config["CoDBand_NoiseScale"]) * erfinv(1.0 - 1e-1)
    config["CoDBand_alpha"] = -1

    realExperiment = experimentOneRealData(namelabel = namelabel,
                        dataset = dataset,
                        context_dimension = config["context_dimension"],
                        plot = plot,
                        Write_to_File = Write_to_File)

    print("Starting for {}, context dimension {}".format(realExperiment.dataset, realExperiment.context_dimension))
    algorithms = {}
    if not args.alg:
        algorithms['UniformLinUCB'] = UniformLinUCB(dimension=config["context_dimension"], alpha=config["LinUCB_alpha"],
                                      lambda_=config["lambda_"], NoiseScale=0.1)
        algorithms['LinUCB'] = LinUCB(dimension=config["context_dimension"], alpha=config["LinUCB_alpha"],
                                      lambda_=config["lambda_"], NoiseScale=0.1)
        algorithms['adTS'] = AdaptiveThompson(dimension=config["context_dimension"], AdTS_Window=config["AdTS_Window"],
                                              AdTS_CheckInter=50, v=config["v"])  # v=0.1
        algorithms['dLinUCB'] = dLinUCB(dimension=config["context_dimension"], alpha=config["dLinUCB_alpha"],
                                        lambda_=config["lambda_"], NoiseScale=config["dLinUCB_NoiseScale"], tau=config["tau"],
                                        delta_1=config["delta_1"], delta_2=config["delta_2"],
                                        tilde_delta_1=config["tilde_delta_1"], eta=config["eta"])
        algorithms['CLUB'] = CLUBAlgorithm(dimension=config["context_dimension"], alpha=config["CLUB_alpha"],
                                           lambda_=config["lambda_"], n=config["n_users"],
                                           alpha_2=config["CLUB_alpha_2"], cluster_init=config["cluster_init"])
        algorithms['SCLUB'] = SCLUB(nu=config["n_users"], d=config["context_dimension"], NoiseScale=config["dLinUCB_NoiseScale"], alpha=config["LinUCB_alpha"], lambda_=config["lambda_"])
        algorithms['CoDBand'] = CoDBand(v = config["CoDBand_v"], d=config["context_dimension"], alpha=config["CoDBand_alpha"] ,lambda_=config["lambda_"], NoiseScale=config["CoDBand_NoiseScale"], alpha_prior=config["alpha_prior"], tau_cd=config["tau"], memory_size=config["memory_size"], delta_1=config["delta_1"], delta_2=config["delta_2"], eta=config["CoDBand_eta"], disable_change_detector=config["disable_change_detector"],true_alpha_0 = config["true_alpha_0"])

    startTime = datetime.datetime.now()
    if dataset == "LastFM":
        address = LastFM_save_address
    elif dataset == "MovieLens":
        address = MovieLens_save_address
    else:
        address = Delicious_save_address
    print(address)

    cfg_path = os.path.join(address, 'Config' + str(namelabel) + startTime.strftime('_%m_%d_%H_%M_%S') + '.json')

    end_num = 0
    while os.path.exists(cfg_path):
        cfg_path = os.path.join(address, 'Config' + str(namelabel) + startTime.strftime('_%m_%d_%H_%M_%S') + str(end_num) + '.json')
        end_num += 1

    with open(cfg_path, 'w') as fp:
        json.dump(config, fp)
    realExperiment.runAlgorithms(algorithms, startTime)