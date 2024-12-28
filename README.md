# PredictiveWorkExample
In this example, we load the input file
- Create features
- Compute Base-line accuracy to compare.
- Divide feature set into train and test dataset
- Define hyper-parameters for SVM model using k-fold technique.
- Select the best model and create classification report for Accuracy

### Load the Libraries
import pandas as pd<br>
import numpy as np<br>
import os<br>
import re<br>

import torch<br>
import torch.nn.functional as F<br>
import matplotlib.pyplot as plt<br>
from sklearn.preprocessing import MinMaxScaler<br>
import seaborn as sns<br>

from tqdm.notebook import tqdm_notebook<br>
tqdm_notebook.pandas()<br>

### Load the feature data file

**Load file with user summaries** 

filename = os.path.join(os.path.dirname(__name__), "DataFiles\\feature_dataset.tsv")<br> 
df = pd.read_csv(filename,delimiter='\t',low_memory=False)<br>
df.head(2)<br>

[[assets/image/Figure-1.PNG]]

### Select the labelled data

**read the summary data**
filename = os.path.join(os.path.dirname(__name__), "DataFiles\\target_labels.csv")<br>
target = pd.read_csv(filename,delimiter=',',low_memory=False)<br>
<br>
labelled_movie_list = list(target.film_id.unique())<br>

#### Create unlabelled dataset

unlabelled_df = df[~df['film_id'].isin(labelled_movie_list)]<br>

#### Merge feature data with target labels

**individual summaries**

single_summary_df = df[df['film_id'].isin(labelled_movie_list)]<br>
single_df = pd.merge(single_summary_df, target, how="left", on=["film_id"])<br>
<br>
#### Pre-process dataset
**lowercase the target labels and strip any extra spaces**

single_df['manual_label'] = single_df['manual_label'].progress_apply(lambda x: (x.lower().strip())) <br>
X = single_df.drop(['manual_label','summary_id', 'summary','film_id',], axis=1)<br>
y = single_df['manual_label']mbda x: (x.lower().strip()))<br>

single_df.manual_label.value_counts()<br>

### Data split into train and test

from sklearn.model_selection import train_test_split<br>

**Splitting the labeled data**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)<br>

print(f"X_train.shape: {X_train.shape}")<br>
print(f"X_test.shape: {X_test.shape}")<br>

### Feature creation
Convert string embeddings into tensor normalized embeddings<br>

**convert strings into normalized embeddings**

X_train['st_embeddings'] = X_train.st_embeddings.progress_apply(lambda x: F.normalize(eval("torch." + x), p=2, dim=1))<br>
X_test['st_embeddings'] = X_test.st_embeddings.progress_apply(lambda x: F.normalize(eval("torch." + x), p=2, dim=1))<br>
X_train.info()<br>

**Normalize sentiment score between [0,1]**

scaler = MinMaxScaler(feature_range=(0, 1))<br>
sentiment_train_scaled = scaler.fit_transform(X_train['sentiment_score'].to_numpy().reshape(-1, 1))<br>
sentiment_test_scaled = scaler.fit_transform(X_test['sentiment_score'].to_numpy().reshape(-1, 1))<br>

**stack embeddings as np.array**

X_train_embeddings = np.vstack(X_train['st_embeddings'].values)<br>
X_test_embeddings = np.vstack(X_test['st_embeddings'].values)<br>
X_train_embeddings.shape<br>

#### Create a feature set with embeddings and sentiment scores

X_train_features = np.column_stack((X_train_embeddings, sentiment_train_scaled))<br>
X_test_features = np.column_stack((X_test_embeddings, sentiment_test_scaled))<br>

### Correct imbalance using class weights

from sklearn.utils.class_weight import compute_class_weight<br>

class_weights = compute_class_weight('balanced', classes = np.unique(y), y = y)<br>
class_weights<br>

class_weight_dict = dict(zip(np.unique(y), class_weights))<br>
class_weight_dict<br>

### Base-line accuracy

from sklearn.dummy import DummyClassifier<br>
from sklearn.metrics import accuracy_score<br>

**Features = embeddings + sentiment scores**

dc1 = DummyClassifier(strategy = 'most_frequent')<br>
dc1.fit(X_train_features, y_train)<br>

acc1 = accuracy_score(y_train, dc1.predict(X_train_features))<br>

print(f'Baseline accuracy = {acc1:.3f}')<br>

### Apply SVM 
from sklearn.svm import SVC<br>
from sklearn.decomposition import PCA<br>
from sklearn.preprocessing import StandardScaler<br>
from sklearn.pipeline import make_pipeline<br>
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold<br>

### SVC models hyper parameters defined

param_C_values = [1, 10, 100, 1000]<br>
param_gamma_values = [0.1, 0.01, 0.001, 0.0001]<br>

param_grid = [<br>
              { 'kernel' : ['linear'],<br>
               'class_weight' : [None, class_weight_dict],<br>
               'C' : param_C_values<br>
               },<br>
              { 'kernel' : ['poly'], <br>
               'C' : param_C_values, <br>
               'class_weight' : [None, class_weight_dict],<br>
               'degree' : [2, 3, 4, 5, 6], <br>
               'gamma' : param_gamma_values<br>
               },<br>
              { 'kernel' : ['rbf'],<br>
               'C' : param_C_values,<br>
               'class_weight' : [None, class_weight_dict],<br>
               'gamma' : param_gamma_values<br>
               }<br>
]<br>

### SVC models with features - embeddings + sentiment score 

from sklearn.model_selection import cross_val_score<br>

inner_cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)<br>
outer_cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)<br>

**inner loop for parameter search**

gs = GridSearchCV(estimator = SVC(),<br>
                  param_grid = param_grid,<br>
                  cv = inner_cv_folds,<br>
                  scoring = 'accuracy',<br>
                  verbose = 0)<br>

**outer loop for accuracy scoring**

scores = cross_val_score(gs, X = X_train_features, y = y_train, cv = outer_cv_folds, verbose = 99)<br>

from sklearn.model_selection import learning_curve<br>

train_sizes, train_scores, test_scores = learning_curve(<br>
    gs, X=X_train_features, y=y_train, cv=outer_cv_folds, <br>
    scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), verbose=0<br>
)<br>

**Calculate mean and standard deviation for plotting**

train_scores_mean = np.mean(train_scores, axis=1)<br>
train_scores_std = np.std(train_scores, axis=1)<br>
test_scores_mean = np.mean(test_scores, axis=1)<br>
test_scores_std = np.std(test_scores, axis=1)<br>

**Plotting the learning curve**

plt.figure(figsize=(10, 6))<br>
plt.title("Learning Curve (SVC)")<br>
plt.xlabel("Training examples")<br>
plt.ylabel("Score")<br>
plt.grid()<br>

**Plot training scores**

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,<br>
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")<br>
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")<br>

**Plot cross-validation scores**

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,<br>
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")<br>
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")<br>

plt.legend(loc="best")<br>
plt.show()<br>

sns.boxplot(x = scores)<br>
sns.swarmplot(x = scores, color = 'black')<br>
plt.title('Model accuracy values (repeated $k$-fold cross validation)')<br>
plt.show()<br>

print(f'Mean score using nested cross-validation: {scores.mean():.3f} +/- {scores.std():.3f}')<br>

cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)<br>

gs = GridSearchCV(estimator = SVC(),<br>
                  param_grid = param_grid,<br>
                  cv = cv_folds,<br>
                  scoring = 'accuracy',<br>
                  return_train_score = True,<br>
                  refit = True,<br>
                  verbose = 3)<br>

#### Running model selection on the training set:

gs.fit(X_train_features, y_train)<br>

print(gs.best_score_)<br>
best_model_params = gs.best_params_<br>
print(best_model_params)<br>

best_model = SVC()<br>
best_model.set_params(**best_model_params)<br>

best_model.probability = True    <br>
best_model.fit(X_train_features, y_train)<br>

#### Evaluating the best model

y_pred = best_model.predict(X_test_features)<br>

from sklearn.metrics import classification_report, ConfusionMatrixDisplay<br>
print(classification_report(y_test, best_model.predict(X_test_features)))<br>

