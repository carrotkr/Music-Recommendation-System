import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#%%
data = pd.read_csv(("/Users/carrotkr/Dropbox/Final/Data_2010s.csv"))
print(data.head())
print(data.describe())

#%%
data['target'].value_counts()
data.columns.values

#%% Data Visualization #1 - Categorical Features [key], [mode], [time_signature].
fig_ctg = plt.figure(figsize=(16,16))

axis = fig_ctg.add_subplot(221)
axis.set_title('key: The estimated overall key of the track.')
data[data['target']==1]['key'].hist(bins=50, label='hit')
data[data['target']==0]['key'].hist(alpha=0.7, bins=50, label='not hit')
plt.legend()

axis = fig_ctg.add_subplot(222)
axis.set_title('mode: The modality (major or minor) of a track.')
data[data['target']==1]['mode'].hist(bins=50, label='hit')
data[data['target']==0]['mode'].hist(alpha=0.7, bins=50, label='not hit')
plt.legend()

axis = fig_ctg.add_subplot(223)
axis.set_title('time_signature: An estimated overall time signature of a track. ')
data[data['target']==1]['time_signature'].hist(bins=50, label='hit')
data[data['target']==0]['time_signature'].hist(alpha=0.7, bins=50, label='not hit')
plt.legend()

#%% Data Visualization #2 - [duration_ms], [acousticness].
fig_1 = plt.figure(figsize=(16,16))

axis = fig_1.add_subplot(221)
axis.set_title('duration_ms: The duration of the track in milliseconds.')
data[data['target']==1]['duration_ms'].hist(bins=50, label='hit')
data[data['target']==0]['duration_ms'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_1.add_subplot(222)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['duration_ms'], fit=norm)
sns.distplot(data[data['target']==0]['duration_ms'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

axis = fig_1.add_subplot(223)
axis.set_title('acousticness: The higher the value the more acoustic the song is.')
data[data['target']==1]['acousticness'].hist(bins=50, label='hit')
data[data['target']==0]['acousticness'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_1.add_subplot(224)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['acousticness'], fit=norm)
sns.distplot(data[data['target']==0]['acousticness'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

#%% Data Visualization #3 - [danceability], [energy].
fig_2 = plt.figure(figsize=(16,16))

axis = fig_2.add_subplot(221)
axis.set_title('danceability: The higher the value, the easier it is to dance to this song.')
data[data['target']==1]['danceability'].hist(bins=50, label='hit')
data[data['target']==0]['danceability'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_2.add_subplot(222)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['danceability'], fit=norm)
sns.distplot(data[data['target']==0]['danceability'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

axis = fig_2.add_subplot(223)
axis.set_title('energy: The higher the value, the more energtic.')
data[data['target']==1]['energy'].hist(bins=50, label='hit')
data[data['target']==0]['energy'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_2.add_subplot(224)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['energy'], fit=norm)
sns.distplot(data[data['target']==0]['energy'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

#%% Data Visualization #4 - [instrumentalness], [liveness].
fig_3 = plt.figure(figsize=(16,16))

axis = fig_3.add_subplot(221)
axis.set_title('instrumentalness: Predicts whether a track contains no vocals.')
data[data['target']==1]['instrumentalness'].hist(bins=50, label='hit')
data[data['target']==0]['instrumentalness'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_3.add_subplot(222)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['instrumentalness'], fit=norm)
sns.distplot(data[data['target']==0]['instrumentalness'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

axis = fig_3.add_subplot(223)
axis.set_title('liveness: The higher the value, the more likely the song is a live recording.')
data[data['target']==1]['liveness'].hist(bins=50, label='hit')
data[data['target']==0]['liveness'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_3.add_subplot(224)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['liveness'], fit=norm)
sns.distplot(data[data['target']==0]['liveness'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

#%% Data Visualization #5 - [speechiness], [valence].
fig_4 = plt.figure(figsize=(16,16))

axis = fig_4.add_subplot(221)
axis.set_title('speechiness: The higher the value the more spoken word the song contains.')
data[data['target']==1]['speechiness'].hist(bins=50, label='hit')
data[data['target']==0]['speechiness'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_4.add_subplot(222)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['speechiness'], fit=norm)
sns.distplot(data[data['target']==0]['speechiness'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

axis = fig_4.add_subplot(223)
axis.set_title('valence: The higher the value, the more positive mood for the song.')
data[data['target']==1]['valence'].hist(bins=50, label='hit')
data[data['target']==0]['valence'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_4.add_subplot(224)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['valence'], fit=norm)
sns.distplot(data[data['target']==0]['valence'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

#%% Data Visualization #6 - [loudness], [tempo].
fig_5 = plt.figure(figsize=(16,16))

axis = fig_5.add_subplot(221)
axis.set_title('loudness(dB): The higher the value, the louder the song.')
data[data['target']==1]['loudness'].hist(bins=50, label='hit')
data[data['target']==0]['loudness'].hist(alpha=0.7, bins=50, label='not hit')
plt.ylabel('count')
plt.legend()

axis = fig_5.add_subplot(222)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['loudness'], fit=norm)
sns.distplot(data[data['target']==0]['loudness'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

axis = fig_5.add_subplot(223)
axis.set_title('tempo(BPM): The overall estimated tempo of a track in beats per minute')
data[data['target']==1]['tempo'].hist(bins=50, label='hit')
data[data['target']==0]['tempo'].hist(alpha=0.7, bins=50, label='not hit')
plt.legend()

axis = fig_5.add_subplot(224)
axis.set_title('(Univariate Distribution)')
sns.distplot(data[data['target']==1]['tempo'], fit=norm)
sns.distplot(data[data['target']==0]['tempo'])
plt.legend(labels=['hit - normal distribution', 'hit', 'not hit'])

#%% Missing Data.
print(data.isnull().sum().sort_values(ascending=False))
print(data.info())

#%%
data.drop(['track', 'artist', 'uri', 'chorus_hit', 'sections'], axis=1, inplace=True)

data_X = data.drop('target', axis=1)
data_Y = data['target']

#%% Standardize Features.
from sklearn.preprocessing import StandardScaler

data_X_std = StandardScaler().fit_transform(data_X)

#%% K-Nearest Neighbors: training, validation, test (60%, 20%, 20%).
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=33)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, Y_train)

print('K-Nearest Neighbors (Training Accuracy):', model_knn.score(X_train, Y_train))

print('K-Nearest Neighbors (Validation Accuracy):', model_knn.score(X_val, Y_val))

Y_predict_knn = model_knn.predict(X_test)
print('K-Nearest Neighbors (Test Accuracy):', accuracy_score(Y_test, Y_predict_knn))

#%% K-Nearest Neighbors: k-fold cross-validation.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, Y_train)

print('K-Nearest Neighbors (Training Accuracy):', model_knn.score(X_train, Y_train))

k_fold = KFold(n_splits=10)
k_fold_score = cross_val_score(model_knn, X_train, Y_train, cv=k_fold)
print('Cross-validation:', k_fold_score)
print('K-Nearest Neighbors (Cross-validation Score Accuracy):', np.mean(k_fold_score))

Y_predict_knn = model_knn.predict(X_test)
print('K-Nearest Neighbors (Test Accuracy):', accuracy_score(Y_test, Y_predict_knn))

#%% K-Nearest Neighbors: k-fold cross-validation, Standardize Features.
X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(data_X_std, data_Y, test_size=0.2, random_state=33)

model_knn_std = KNeighborsClassifier()
model_knn_std.fit(X_train_std, Y_train_std)

print('K-Nearest Neighbors (Training Accuracy):', model_knn_std.score(X_train_std, Y_train_std))

k_fold = KFold(n_splits=10)
k_fold_score_std = cross_val_score(model_knn_std, X_train_std, Y_train_std, cv=k_fold)
print('Cross-validation: ', k_fold_score_std)
print('K-Nearest Neighbors (Cross-validation Score Accuracy):', np.mean(k_fold_score_std))

Y_predict_knn_std = model_knn_std.predict(X_test_std)
print('K-Nearest Neighbors (Test Accuracy):', accuracy_score(Y_test_std, Y_predict_knn_std))

#%% Logistic Regression: training, validation, test (60%, 20%, 20%).
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=33)

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)

print('Logistic Regression (Training Accuracy):', model_lr.score(X_train, Y_train))

print('Logistic Regression (Validation Accuracy):', model_lr.score(X_val, Y_val))

Y_predict_lr = model_lr.predict(X_test)
print('Logistic Regression (Test Accuracy):', accuracy_score(Y_test, Y_predict_lr))

#%% Logistic Regression: L1 regularization.
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=33)

model_lr_l1 = LogisticRegression(penalty='l1')
model_lr_l1.fit(X_train, Y_train)

print('Logistic Regression (Training Accuracy):', model_lr_l1.score(X_train, Y_train))

print('Logistic Regression (Validation Accuracy):', model_lr_l1.score(X_val, Y_val))

Y_predict_lr_l1 = model_lr_l1.predict(X_test)
print('Logistic Regression (Test Accuracy):', accuracy_score(Y_test, Y_predict_lr_l1))

#%% Logistic Regression: Standardize Features.
X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(data_X_std, data_Y, test_size=0.2, random_state=33)
X_train_std, X_val_std, Y_train_std, Y_val_std = train_test_split(X_train_std, Y_train_std, test_size=0.25, random_state=33)

model_lr_std = LogisticRegression()
model_lr_std.fit(X_train_std, Y_train_std)

print('Logistic Regression (Training Accuracy):', model_lr_std.score(X_train_std, Y_train_std))

print('Logistic Regression (Validation Accuracy):', model_lr_std.score(X_val_std, Y_val_std))

Y_predict_lr_std = model_lr_std.predict(X_test_std)
print('Logistic Regression (Test Accuracy):', accuracy_score(Y_test_std, Y_predict_lr_std))

#%% Decision Tree: training, validation, test (60%, 20%, 20%).
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=33)

from sklearn.tree import DecisionTreeClassifier

model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(X_train, Y_train)

print('Decision Tree (Training Accuracy):', model_decision_tree.score(X_train, Y_train))

print('Decision Tree (Validation Accuracy):', model_decision_tree.score(X_val, Y_val))

Y_predict_decision_tree = model_decision_tree.predict(X_test)
print('Decision Tree (Test Accuracy):', accuracy_score(Y_test, Y_predict_decision_tree))

#%% Decision Tree: Standardize Features.
X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(data_X_std, data_Y, test_size=0.2, random_state=33)
X_train_std, X_val_std, Y_train_std, Y_val_std = train_test_split(X_train_std, Y_train_std, test_size=0.25, random_state=33)

model_decision_tree_std = DecisionTreeClassifier()
model_decision_tree_std.fit(X_train_std, Y_train_std)

print('Decision Tree (Training Accuracy):', model_decision_tree_std.score(X_train_std, Y_train_std))

print('Decision Tree (Validation Accuracy):', model_decision_tree_std.score(X_val_std, Y_val_std))

Y_predict_decision_tree_std = model_decision_tree_std.predict(X_test_std)
print('Decision Tree (Test Accuracy):', accuracy_score(Y_test_std, Y_predict_decision_tree_std))

#%% Random Forest: training, validation, test (60%, 20%, 20%).
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=33)

from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier(n_estimators=100)
model_random_forest.fit(X_train, Y_train)

print('Random Forest (Training Accuracy):', model_random_forest.score(X_train, Y_train))

print('Random Forest (Validation Accuracy):', model_random_forest.score(X_val, Y_val))

Y_predict_random_forest = model_random_forest.predict(X_test)
print('Random Forest (Test Accuracy):', accuracy_score(Y_test, Y_predict_random_forest))

#%% Lasso Regression.
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=33)

from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(X_train,Y_train)
print('alpha=0.01')
print('Lasso Score: ', model_lasso.score(X_train, Y_train))
print('Used number of coefficients: ', np.sum(model_lasso.coef_ != 0))

model_lasso = Lasso(alpha=0.001)
model_lasso.fit(X_train,Y_train)
print('alpha=0.001')
print('Lasso Score: ', model_lasso.score(X_train, Y_train))
print('Used number of coefficients: ', np.sum(model_lasso.coef_ != 0))

model_lasso = Lasso(alpha=0.0001)
model_lasso.fit(X_train,Y_train)
print('alpha=0.0001')
print('Lasso Score: ', model_lasso.score(X_train, Y_train))
print('Used number of coefficients: ', np.sum(model_lasso.coef_ != 0))