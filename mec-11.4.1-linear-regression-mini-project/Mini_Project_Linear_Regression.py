%reset -f

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as stats
import sklearn

os.chdir(r'C:\UCSD_ML\mec-mini-projects\mec-11.4.1-linear-regression-mini-project')

sns.set_style("whitegrid")
sns.set_context("poster")

########## 2 ##########
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()

bos = pd.DataFrame(boston.data)
bos.head()
bos.columns = boston.feature_names

bos['PRICE'] = pd.Series(boston.target)


plt.scatter(np.log(bos.CRIM), np.log(bos.PRICE), s=1)
plt.show()



from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
# X = bos.drop('PRICE', axis=1)
X = np.array(bos['PTRATIO']).reshape(-1,1)
y = bos['PRICE']
lm.fit(X, y)

print(lm.coef_)
print(lm.intercept_)

lm.score(X,y)

RSS = (np.sum((bos.PRICE - lm.predict(X)) ** 2))
ESS = (np.sum((lm.predict(X) - np.mean(bos.PRICE))**2))

K=1; N=len(y)
MSR = ESS/K
MSE = RSS/(N-K-1)






# Your turn.
# your turn
# your turn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
# X = bos.drop('PRICE', axis=1)
X = np.array(bos[['CRIM', 'RM','PTRATIO']])
y = bos['PRICE']
lm.fit(X, y)

y_fit = lm.predict(X)
res = y-y_fit
plt.scatter(y_fit, res, s=5)
plt.xlabel('y_fit')
plt.ylabel('residual')
plt.title('y_fit vs residual')
plt.show()



stats.probplot(res, dist="norm", plot=plt)
plt.show()


print(lm.coef_)
print(lm.intercept_)

# R2
print(lm.score(X,y))

# F
RSS = (np.sum((bos.PRICE - lm.predict(X)) ** 2))
ESS = (np.sum((lm.predict(X) - np.mean(bos.PRICE))**2))
K=1; N=len(y)
MSR = ESS/K
MSE = RSS/(N-K-1)
print(MSR/MSE)



plt.scatter(y, y_fit)

res[np.abs(res) > 3*np.std(res)]

stats.norm.ppf([.005,0.1,09])





