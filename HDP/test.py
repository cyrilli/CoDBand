import numpy as np

# a = [1,2,3,4]
# print((np.append(a,5)))
# print(len(np.append(a,5)))
#
# print(np.delete(a, 1))
# for i in range(30):
# 	print(np.random.choice(4))

# X_new = np.array([[1,1,1,2], [1,1,1,1]]).reshape([-1,4])
# #
# # print(np.dot(X_new, np.transpose(X_new)))
#
# x = [ 0.003658,    0.16609211, -0.22520198, -0.08061512, -0.18144348, -0.25761604,
#   0.36162504  ,0.15076518  ,0.18868475 , 0.47092  ,   0.09683141, -0.20430542,
#  -0.16155256 , 0.42209016 ,-0.03352851 , 0.03507998  ,0.2502606,  -0.1017855,
#  -0.13370293 ,-0.14288846 , 0.07871718, -0.14655584 ,-0.00102108, -0.03380401,
#   0.08481165]
# x = np.array(x)
# Covariance = np.linalg.inv(0.1 * np.identity(25))
# sigma = 0.1
# print(np.matmul(np.matmul(x.transpose(), Covariance), x))
#
# print(Covariance)
pp = 600
pp = np.exp(pp)  # - np.max(pp))  # TODO: why?
pp = pp / np.sum(pp)