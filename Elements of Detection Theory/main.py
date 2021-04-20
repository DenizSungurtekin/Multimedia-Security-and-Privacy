import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

u = [0,1,2]
p0 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
p1 = p0[::-1]

def computeTau(u,p0,p1):
    return np.log(p1/p0) * (2 + u)/-2

def computeProbFalse(tau):
    return 1 - norm.cdf(tau)

def computeProbMiss(tau,u):
    return norm.cdf(tau-u)


for element in u:
    p_m = [computeProbMiss(computeTau(element,p0[i],p1[i]),element) for i in range(len(p0))]
    p_f = [computeProbFalse(computeTau(element, p0[i], p1[i])) for i in range(len(p0))]
    plt.plot(p_f,p_m,label = element)



plt.xlabel("pf")
plt.ylabel("pm")
plt.title("ROC")
plt.legend()
plt.show()

