# PredictiveWorkExample

### Load the Libraries
import pandas as pd<br>
import numpy as np
import os
import re

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()

### Load the feature data file

# Load file with user summaries
filename = os.path.join(os.path.dirname(__name__), "DataFiles\\feature_dataset.tsv")
df = pd.read_csv(filename,delimiter='\t',low_memory=False)
df.head(2)

### Select the labelled data

# read the summary data
filename = os.path.join(os.path.dirname(__name__), "DataFiles\\target_labels.csv")
target = pd.read_csv(filename,delimiter=',',low_memory=False)

labelled_movie_list = list(target.film_id.unique())

#### Create unlabelled dataset

unlabelled_df = df[~df['film_id'].isin(labelled_movie_list)]

#### Merge feature data with target labels

# individual summaries
single_summary_df = df[df['film_id'].isin(labelled_movie_list)]
single_df = pd.merge(single_summary_df, target, how="left", on=["film_id"])

#### Pre-process dataset
lowercase the target labels and strip any extra spaces

single_df['manual_label'] = single_df['manual_label'].progress_apply(la
X = single_df.drop(['manual_label','summary_id', 'summary','film_id',], axis=1)
y = single_df['manual_label']mbda x: (x.lower().strip()))

single_df.manual_label.value_counts()

### Data split into train and test

from sklearn.model_selection import train_test_split

# Splitting the labeled data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")

### Feature creation
Convert string embeddings into tensor normalized embeddings

# convert strings into normalized embeddings
X_train['st_embeddings'] = X_train.st_embeddings.progress_apply(lambda x: F.normalize(eval("torch." + x), p=2, dim=1))
X_test['st_embeddings'] = X_test.st_embeddings.progress_apply(lambda x: F.normalize(eval("torch." + x), p=2, dim=1))
X_train.info()

#  Normalize sentiment score between [0,1]

scaler = MinMaxScaler(feature_range=(0, 1))
sentiment_train_scaled = scaler.fit_transform(X_train['sentiment_score'].to_numpy().reshape(-1, 1))
sentiment_test_scaled = scaler.fit_transform(X_test['sentiment_score'].to_numpy().reshape(-1, 1))

# stack embeddings as np.array
X_train_embeddings = np.vstack(X_train['st_embeddings'].values)
X_test_embeddings = np.vstack(X_test['st_embeddings'].values)
X_train_embeddings.shape

#### Create a feature set with embeddings and sentiment scores

X_train_features = np.column_stack((X_train_embeddings, sentiment_train_scaled))
X_test_features = np.column_stack((X_test_embeddings, sentiment_test_scaled))

### Correct imbalance using class weights

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes = np.unique(y), y = y)
class_weights

class_weight_dict = dict(zip(np.unique(y), class_weights))
class_weight_dict

### Base-line accuracy

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Features = embeddings + sentiment scores
dc1 = DummyClassifier(strategy = 'most_frequent')
dc1.fit(X_train_features, y_train)

acc1 = accuracy_score(y_train, dc1.predict(X_train_features))

print(f'Baseline accuracy = {acc1:.3f}')

### Apply SVM 
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

### SVC models hyper parameters defined

param_C_values = [1, 10, 100, 1000]
param_gamma_values = [0.1, 0.01, 0.001, 0.0001]

param_grid = [
              { 'kernel' : ['linear'],
               'class_weight' : [None, class_weight_dict],
               'C' : param_C_values
               },
              { 'kernel' : ['poly'], 
               'C' : param_C_values, 
               'class_weight' : [None, class_weight_dict],
               'degree' : [2, 3, 4, 5, 6], 
               'gamma' : param_gamma_values
               },
              { 'kernel' : ['rbf'],
               'C' : param_C_values,
               'class_weight' : [None, class_weight_dict],
               'gamma' : param_gamma_values
               }
]

### SVC models with features - embeddings + sentiment score 

from sklearn.model_selection import cross_val_score

inner_cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)
outer_cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)

# inner loop for parameter search
gs = GridSearchCV(estimator = SVC(),
                  param_grid = param_grid,
                  cv = inner_cv_folds,
                  scoring = 'accuracy',
                  verbose = 0)

# outer loop for accuracy scoring
scores = cross_val_score(gs, X = X_train_features, y = y_train, cv = outer_cv_folds, verbose = 99)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    gs, X=X_train_features, y=y_train, cv=outer_cv_folds, 
    scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), verbose=0
)

# Calculate mean and standard deviation for plotting
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning Curve (SVC)")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

# Plot training scores
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# Plot cross-validation scores
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()


sns.boxplot(x = scores)
sns.swarmplot(x = scores, color = 'black')
plt.title('Model accuracy values (repeated $k$-fold cross validation)')
plt.show()


print(f'Mean score using nested cross-validation: {scores.mean():.3f} +/- {scores.std():.3f}')

cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)

gs = GridSearchCV(estimator = SVC(),
                  param_grid = param_grid,
                  cv = cv_folds,
                  scoring = 'accuracy',
                  return_train_score = True,
                  refit = True,
                  verbose = 3)

#### Running model selection on the training set:

gs.fit(X_train_features, y_train)

print(gs.best_score_)
best_model_params = gs.best_params_
print(best_model_params)

best_model = SVC()
best_model.set_params(**best_model_params)

best_model.probability = True    
best_model.fit(X_train_features, y_train)

#### Evaluating the best model

best_model.score(X_test_features, y_test)

y_pred = best_model.predict(X_test_features)

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(y_test, best_model.predict(X_test_features)))
