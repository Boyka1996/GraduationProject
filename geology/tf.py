import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
#np.random.seed(0)
# x = np.random.randn(80, 2)
# y = x[:, 0] + 2*x[:, 1] + np.random.randn(80)

df = pd.read_table("ARRAY LATEROLOG.TXT",header=None,sep=r'\s{2,}',engine='python')#利用正则表达式处理txt的空格，并设置python解释正则
x1 = df.iloc[0:,0]
y1 = df.iloc[0:,2]
x = np.array(x1).reshape(-1,1)
y = np.array(y1)
# print(y)
# print(y1)
clf = SVR(kernel='linear', C=1.2)
x_tran,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
plt.plot(x_tran, y_train, 'go-', label="predict")
plt.plot(x_test, y_test, 'co-', label="real")

print(x_tran)
clf.fit(x_tran, y_train)
y_hat = clf.predict(x_test)
print("得分:", r2_score(y_test, y_hat))
r = len(x_test) + 1
print(y_test)
plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
plt.plot(np.arange(1,r), y_test, 'co-', label="real")
plt.legend()
plt.show()

