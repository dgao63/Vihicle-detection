import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.grid_search import RandomizedSearchCV
print("modules loaded")


data = pickle.load( open( "split_data.p", "rb" ))
X_train = data["X_train"]
y_train = data["y_train"]
X_validation = data["X_validation"]
y_validation = data["y_validation"]
X_test = data["X_test"]
y_test = data["y_test"]

print(len(X_train))
print(len(X_validation))
print(len(X_test))
#X = np.vstack((car_features, notcar_features)).astype(np.float32)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_validation = X_scaler.transform(X_validation)
X_test = X_scaler.transform(X_test)


alg1 = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
        probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
        max_iter=-1, decision_function_shape=None, random_state=None)
alg2 = RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=None, min_samples_split=50)


#clf = VotingClassifier(estimators=[('svm', alg1), ('rf', alg2)], voting='soft')
clf = alg1
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
score = clf.score(X_validation, y_validation)
print("accuracy:", score)
print(round(t2-t, 2), 'Seconds to tune SVC...')

todump = {"scaler":X_scaler, "clf":clf}
pickle.dump(todump, open("model_scaler.p", 'wb'))

print(clf.predict_proba(X_test[0]))
print(clf.predict_proba(X_test))



'''
def svm_param_tuning(): 
    tuning_1 = svm.SVC()
    svm_param = {   "C": [0.001, 0.01, 0.1, 1]
                }
    random_search = RandomizedSearchCV(tuning_1, param_distributions=svm_param,
                                   n_iter=3)
    random_search.fit(X_train, y_train)
    print("the best svm parameters:")
    print(random_search.best_estimator_)
    print("the parameters have the score of:")
    print(random_search.best_score_)
#C=1 acc:0.91
t=time.time()
svm_param_tuning()
t2 = time.time()
print(round(t2-t, 2), 'Seconds to tune SVC...')


print(np.rint(np.linspace(X_train.shape[0]*2, X_train.shape[0]*4, 5)).astype(int))
print(np.rint(np.linspace(1, X_train.shape[1]/2, 5)).astype(int))
print(np.rint(np.linspace(2, X_train.shape[0]/50, 5)).astype(int))
def forest_param_tuning():
    tuning_2 = RandomForestClassifier()
    sqrtfeat = np.sqrt(X_train.shape[1])
    forest_param = { "n_estimators"      : [2000],
                    "min_samples_split" : [10, 100]
                    }
    random_search = RandomizedSearchCV(tuning_2, param_distributions=forest_param,
                                   n_iter=2)
    random_search.fit(X_train, y_train)
    print("the best forest parameters:")
    print(random_search.best_estimator_)
    print("the parameters have the score of:")
    print(random_search.best_score_)

t=time.time()
forest_param_tuning()
t2 = time.time()
print(round(t2-t, 2), 'Seconds to tune random forest...')


def MLP_param_tuning():
    tuning_3 = MLPClassifier()
    forest_param = { "hidden_layer_sizes"      : [1000, 5000, 10000],
                 "alpha"         : [0.0001,0.001]}
    random_search = RandomizedSearchCV(tuning_3, param_distributions=forest_param,
                                   n_iter=4)
    random_search.fit(titanic[predictors], titanic["Survived"])
    print("the best MLP parameters:")
    print(random_search.best_estimator_)
    print("the parameters have the score of:")
    print(random_search.best_score_)

t=time.time()
MLP_param_tuning()
t2 = time.time()
print(round(t2-t, 2), 'Seconds to tune random forest...')
'''
