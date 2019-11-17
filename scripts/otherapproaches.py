from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
def run_algo(x,y,x_test,y_test,algo="ada"):
    if(algo=="ada"):
        clf = AdaBoostClassifier(n_estimators=200000,learning_rate=0.0000001)
    elif(algo=="grad"):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,
                max_depth=20000, random_state=0)
    else:
        print("Please choose appropriate algo, not {0}".format(algo))
        return
    clf.fit(x,y)
    score = clf.score(x,y)
    score_test = clf.score(x_test,y_test)
    print("train",score)
    print("test",score_test)

def run_algo_permuted(x,y,x_test,y_test,algo="ada",num_iteration=10):
    if(algo=="ada"):
        clf = AdaBoostClassifier(n_estimators=200000,learning_rate=0.0000000000000000001)
    elif(algo=="grad"):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,
                max_depth=20000, random_state=0)
    else:
        print("Please choose appropriate algo, not {0}".format(algo))
        return
    x = np.array(x)
    y = np.array(y)
    for i in range(0,num_iteration):
        train = np.random.choice(range(len(x)),6,replace=False)
        x_train = x[train]
        y_train = y[train]
        clf.fit(x_train,y_train)
    print(clf.score(x,y))
    print(clf.score(x_test,y_test))
    

