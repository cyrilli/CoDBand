import numpy as np
from scipy.stats import chi2, multivariate_normal, norm
from scipy.special import erfinv
import pdb, traceback, sys
import time

class GlobalModelStruct:
	def __init__(self, d, lambda_, sigma):
		self.XTX = np.zeros([d, d])
		self.Xy = np.zeros(d)

		self.Covariance = np.linalg.inv(lambda_ * np.identity(n=d) + self.XTX / sigma**2)
		self.Mean = np.dot(self.Covariance, self.Xy / sigma**2)

		self.numDataPoint = 0
		# self.v = sigma * np.sqrt(9*d*np.log(self.numDataPoint / delta))

		self.d = d
		self.lambda_ = lambda_
		self.sigma = sigma
		# self.delta = delta

	def removeUserSS(self, ADelta, bDelta, numDataPointDelta):
		self.XTX -= ADelta
		self.Xy -= bDelta
		self.Covariance = np.linalg.inv(self.lambda_ * np.identity(n=self.d) + self.XTX / self.sigma**2)
		self.Mean = np.dot(self.Covariance, self.Xy / self.sigma**2)

		self.numDataPoint -= numDataPointDelta
		# self.v = self.sigma * np.sqrt(9 * self.d * np.log(self.numDataPoint / self.delta))

	def addUserSS(self, ADelta, bDelta, numDataPointDelta):
		self.XTX += ADelta
		self.Xy += bDelta
		self.Covariance = np.linalg.inv(self.lambda_ * np.identity(n=self.d) + self.XTX / self.sigma**2)
		self.Mean = np.dot(self.Covariance, self.Xy / self.sigma**2)

		self.numDataPoint += numDataPointDelta
		# self.v = self.sigma * np.sqrt(9 * self.d * np.log(self.numDataPoint / self.delta))

	def log_predictive(self, X_new, y_new):
		assert X_new.shape[1] == self.d
		# compute the log posterior predictive distribution evaluating at (X, y)
		num_x_new = X_new.shape[0]
		mean = np.dot(X_new, self.Mean)
		cov = self.sigma**2 * np.identity(n=num_x_new)+ np.dot(X_new, np.dot(self.Covariance, np.transpose(X_new)))
		return multivariate_normal.logpdf(y_new, mean=mean, cov=cov, allow_singular=True)

	# def log_predictive(self, data_x, data_y):
	# 	num_data = len(data_x)
	# 	assert len(data_y) == num_data
	# 	assert data_x.shape == (num_data, self.d)
	# 	assert data_y.shape == (num_data,)
	# 	sum_logpdf = 0
	# 	for data_i in range(num_data):
	# 		var = self.sigma**2 + np.matmul(np.matmul(data_x[data_i].transpose(), self.Covariance), data_x[data_i])
	# 		logpdf_i = norm.logpdf(data_y[data_i], loc=np.dot(data_x[data_i], self.Mean), scale=np.sqrt(var))
	# 		try:
	# 			assert not np.isnan(logpdf_i)
	# 		except:
	# 			type, value, tb = sys.exc_info()
	# 			traceback.print_exc()
	#
	# 		sum_logpdf +=logpdf_i
	# 	return sum_logpdf

	def sampleTheta(self, v=None):
		# if v is None:
		# 	v = self.v
		theta = np.random.multivariate_normal(mean=self.Mean, cov=self.Covariance)
		return theta


class UserModelStruct:
	def __init__(self, userID, dimension, alpha, lambda_, NoiseScale, delta_1, delta_2, createTime, eta, change_detection_alpha=0.01, memory_size=50):
		self.userID = userID
		self.d = dimension
		self.alpha = alpha  # use constant alpha, instead of the one defined in LinUCB
		self.lambda_ = lambda_
		self.change_detection_alpha = change_detection_alpha
		self.delta_1 = delta_1
		self.delta_2 = delta_2

		# LinUCB statistics
		self.A = np.zeros([self.d, self.d])
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A+lambda_ * np.identity(n=self.d))

		self.NoiseScale = NoiseScale
		self.update_num = 0  # number of times this user has been updated
		self.alpha_t = self.NoiseScale * np.sqrt(
			self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
			self.lambda_)

		self.memory_size = memory_size
		# history data
		self.X = np.zeros((0, self.d))
		self.y = np.zeros((0,))

		self.UserTheta = np.zeros(self.d)
		self.UserThetaNoReg = np.zeros(self.d)

		self.rank = 0

		self.createTime = createTime        # global time when model is created
		self.time = 0                       # number of times this user has been served

		# for dLinUCB's change detector
		self.eta = eta  # upper bound of gaussian noise
		self.detectedChangePoints = [0]
		self.failList = []

	def resetLocalUserModel(self, createTime):
		self.outDated = False
		self.A = np.zeros([self.d, self.d])
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A+self.lambda_ * np.identity(n=self.d))
		self.UserTheta = np.zeros(self.d)
		self.UserThetaNoReg = np.zeros(self.d)
		self.X = np.zeros((0, self.d))
		self.rank = 0
		self.y = np.zeros((0,))
		self.update_num = 0  # number of times this user has been updated
		self.alpha_t = self.NoiseScale * np.sqrt(
			self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
			self.lambda_)

		self.createTime = createTime        # global time when model is created
		self.failList = []

	def updateLocalUserModel(self, articlePicked_FeatureVector, click):
		# update LinUCB statistics
		self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector * click
		self.AInv = np.linalg.inv(self.A + self.lambda_ * np.identity(n=self.d))
		self.UserTheta = np.dot(self.AInv, self.b)
		self.UserThetaNoReg = np.dot(np.linalg.pinv(self.A), self.b)
		assert self.d == articlePicked_FeatureVector.shape[0]

		self.update_num += 1.0

		# update observation history
		self.X = np.concatenate((self.X, articlePicked_FeatureVector.reshape(1, self.d)), axis=0)
		self.y = np.concatenate((self.y, np.array([click])),axis=0)
		if len(self.X) > self.memory_size:
			self.X = self.X[-self.memory_size:]
			self.y = self.y[-self.memory_size:]
		# assert self.X.shape == (self.update_num, self.d)
		# assert self.y.shape == (self.update_num, )
		self.rank = np.linalg.matrix_rank(self.X)
		self.alpha_t = self.NoiseScale * np.sqrt(
			self.d * np.log(1 + self.update_num / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_1)) + np.sqrt(
			self.lambda_)

	def getCB(self, x, useConstantAlpha = False):
		var = np.sqrt(np.dot(np.dot(x, self.AInv), x))
		if useConstantAlpha:
			return self.alpha * var
		else:
			return self.alpha_t * var

	def getInstantaneousBadness(self, articlePicked, click, method="ChiSquare"):
		# compute badness on (articlePicked, click)
		if method == "ConfidenceBound":  # This is the test statistic used in dLinUCB
			mean = np.dot(self.UserTheta, articlePicked.contextFeatureVector[:self.d])
			rewardEstimationError = np.abs(mean - click)
			if rewardEstimationError <= self.getCB(articlePicked.contextFeatureVector[:self.d]) + self.eta:
				e = 0
			else:
				e = 1
		elif method == "ChiSquare":
			if self.rank < self.d:
				e = 0
			else:
				x = articlePicked.contextFeatureVector[:self.d]
				if self.rank < self.d:
					e = 0
				else:
					mean = np.dot(self.UserThetaNoReg, x)
					rewardEstimationError = (mean - click)**2
					rewardEstimationErrorSTD = self.NoiseScale**2 * (1 + np.dot(np.dot(x, np.linalg.pinv(self.A)), x))
					df1 = 1

					chiSquareStatistic = rewardEstimationError / rewardEstimationErrorSTD
					p_value = chi2.sf(x=chiSquareStatistic, df=df1)
					if p_value <= self.change_detection_alpha:  # upper bound probability of false alarm
						e = 1
					else:
						e = 0
		# Update failList
		self.failList.append(e)
		return e

	def detectChangeBasedOnBadness(self, ObservationInterval):
		if len(self.failList) < ObservationInterval:
			ObservationNum = float(len(self.failList))
			badness = sum(self.failList) / ObservationNum
		else:
			ObservationNum = float(ObservationInterval)
			badness = sum(self.failList[-ObservationInterval:]) / ObservationNum
		badness_CB = np.sqrt(np.log(1.0 / self.delta_2) / (2.0 * ObservationNum))
		# test badness against threshold
		if badness > self.delta_1 + badness_CB:
			changeFlag = 1
		else:
			changeFlag = 0
		return changeFlag

class CoDBand:
	def __init__(self, d, lambda_, NoiseScale, alpha_prior, tau_cd=25, disable_change_detector=False):
		self.globalModels = []
		self.globalModelPeriodCounter = []  # num of stationary periods associated with each global model
		self.totalPeriodCounter = 0  # total num of stationary periods

		self.userID2globalModelIndex = {}
		self.userModels = {}  # userID : UserModelStruct

		self.dimension = d
		self.lambda_ = lambda_
		self.NoiseScale = NoiseScale

		## params for DP mixture
		self.alpha_prior = alpha_prior
		self.alpha_0 = 1
		# np.random.gamma(self.alpha_prior['a'],self.alpha_prior['b'])

		## Params for change detector
		self.disable_change_detector = disable_change_detector  # disable change detector
		self.tau_cd = tau_cd
		# only used if we detect change using dLinUCB's test statistic
		self.eta = np.sqrt(2.0) * (self.NoiseScale) * erfinv(1.0 - 1e-1)

		self.global_time = 0

		self.CanEstimateUserPreference = True
		self.CanEstimateUserCluster = True

	def decide(self, pool_articles, userID):
		# print("==== arm selection ====")
		t = time.time()
		if userID not in self.userModels:
			# initialize user model struct for the new user
			self.userModels[userID] = UserModelStruct(userID, self.dimension, 0.3, self.lambda_, self.NoiseScale,
													  delta_1=1e-1, delta_2=1e-1, createTime=self.global_time,
													  eta=self.eta)
		# print("time taken for sampling model {}".format(t-time.time()))
		t = time.time()
		if self.userModels[userID].update_num == 0:
			# sample a model index for the new user or the old user that has been detected to have changed
			# either way they represent the beginning of a new stationary period
			self.sample_z(userID)
		# print("time taken for sampling theta {}".format(t - time.time()))

		self.cluster = []
		for k, v in self.userID2globalModelIndex.items():
			if v == self.userID2globalModelIndex[userID]:
				self.cluster.append(self.userModels[k])

		maxPTA = float('-inf')
		articlePicked = None

		thetaTilde = self.globalModels[self.userID2globalModelIndex[userID]].sampleTheta()
		for x in pool_articles:
			x_pta = np.dot(thetaTilde, x.contextFeatureVector)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.global_time += 1
		self.userModels[userID].time += 1
		# print("==== update param ====")
		t = time.time()
		self.userModels[userID].getInstantaneousBadness(articlePicked, click, method="ChiSquare")  # "ChiSquare" or "ConfidenceBound"
		# if e == 0:  # if model is admissible for this observation
		self.userModels[userID].updateLocalUserModel(articlePicked.contextFeatureVector, click)
		self.globalModels[self.userID2globalModelIndex[userID]].addUserSS(
			np.outer(articlePicked.contextFeatureVector, articlePicked.contextFeatureVector),
			articlePicked.contextFeatureVector * click, 1)

		# print("time taken for ss update {}".format(t-time.time()))
		t = time.time()
		# Collapsed Gibbs Sampler for global model index z and alpha_0
		self.sample_z(userID)  # two options: 1) only sample z for current user; 2) sample z for all users
		self.sample_alpha_0()
		# print("time taken for Gibbs sampling {}".format(t - time.time()))
		if self.disable_change_detector:
			changeFlag = False
		else:
			changeFlag = self.userModels[userID].detectChangeBasedOnBadness(self.tau_cd)

		if changeFlag:
			# reset user model
			self.userModels[userID].resetLocalUserModel(self.global_time)
			self.userModels[userID].detectedChangePoints.append(self.userModels[userID].time)

	def sample_z(self, userID):

		if self.userModels[userID].update_num == 0:
			# if new user model
			# sample model index according to the popularity/frequency of global models
			# this is for new user or user that has just changed
			pp = np.log(np.append(self.globalModelPeriodCounter, self.alpha_0))
			pp = np.exp(pp) # - np.max(pp))
			pp = pp / np.sum(pp)
			z = np.random.choice(len(self.globalModels)+1, size=None, replace=True, p=pp)

			if z == len(self.globalModels):
				# new model is sampled
				new_global_model = GlobalModelStruct(d=self.dimension, lambda_=self.lambda_, sigma=self.NoiseScale)
				# print("self.globalModels {}".format(self.globalModels))
				self.globalModels.append(new_global_model)
				self.globalModelPeriodCounter.append(0)

			self.userID2globalModelIndex[userID] = z
			self.globalModelPeriodCounter[z] += 1
			self.totalPeriodCounter += 1

		else:
			# if not new user model, collapsed Gibbs sampler for model index
			temp_z = self.userID2globalModelIndex[userID]
			# remove counter and sufficient statistics (ss) of userID's current period
			self.globalModels[temp_z].removeUserSS(self.userModels[userID].A, self.userModels[userID].b, self.userModels[userID].update_num)
			self.globalModelPeriodCounter[temp_z] -= 1
			self.totalPeriodCounter -= 1

			if self.globalModelPeriodCounter[temp_z] == 0:
				# remove this global model
				# self.globalModels = np.delete(self.globalModels, temp_z)
				del self.globalModels[temp_z]
				# self.globalModelPeriodCounter = np.delete(self.globalModelPeriodCounter, temp_z)
				del self.globalModelPeriodCounter[temp_z]
				# update the model index in userID2globalModelIndex (decrement by 1)
				for k, v in self.userID2globalModelIndex.items():
					if v > temp_z:
						self.userID2globalModelIndex[k] = v-1
					elif v == temp_z:
						# assert: no other user has temp_z as its model index
						assert k == userID

			pp = np.log(np.append(self.globalModelPeriodCounter, self.alpha_0))
			# print("pop log pp {}".format(pp))
			for k in range(0, len(self.globalModels)):
				pp[k] = pp[k] + self.globalModels[k].log_predictive(self.userModels[userID].X, self.userModels[userID].y)
			pp[len(self.globalModels)] += self.log_predictive_by_prior(self.userModels[userID].X, self.userModels[userID].y)

			# print("log pp {}".format(pp))
			# print("==========")
			pp = np.exp(pp- np.max(pp))  # TODO: why?
			pp = pp / np.sum(pp)

			# print("exp pp {}".format(pp))
			# print("len(self.globalModels) + 1 {}".format(len(self.globalModels) + 1))

			z = np.random.choice(len(self.globalModels) + 1, size=None, replace=True, p=pp)

			if z == len(self.globalModels):
				# new model is sampled
				new_global_model = GlobalModelStruct(d=self.dimension, lambda_=self.lambda_, sigma=self.NoiseScale)
				# print("self.globalModels {}".format(self.globalModels))
				self.globalModels.append(new_global_model)
				self.globalModelPeriodCounter.append(0)

			self.userID2globalModelIndex[userID] = z
			self.globalModelPeriodCounter[z] += 1
			self.totalPeriodCounter += 1

			self.globalModels[z].addUserSS(self.userModels[userID].A, self.userModels[userID].b, self.userModels[userID].update_num)

	def sample_alpha_0(self):
		# Escobar and West 1995
		eta = np.random.beta(self.alpha_0 + 1, self.totalPeriodCounter, 1)
		# Teh HDP 2005
		# construct the mixture model
		pi = self.totalPeriodCounter / self.alpha_0
		pi = pi / (1 + pi)
		s = np.random.binomial(1, pi, 1)
		# sample from a two gamma mixture models
		self.alpha_0 = np.random.gamma(self.alpha_prior['a'] + len(self.globalModels) - s, 1 / (self.alpha_prior['b'] - np.log(eta)), 1)
		# print(self.alpha_0)

	def log_predictive_by_prior(self, X_new, y_new):
		assert X_new.shape[1] == self.dimension
		# compute the log prior predictive distribution evaluating at (X, y)
		num_x_new = X_new.shape[0]
		return multivariate_normal.logpdf(y_new, mean=np.zeros(num_x_new), cov=self.NoiseScale**2 * np.identity(n=num_x_new)+ np.dot(X_new, np.transpose(X_new))/self.lambda_)


	def getTheta(self, userID):
		return self.globalModels[self.userID2globalModelIndex[userID]].Mean

if __name__ == '__main__':
	pass