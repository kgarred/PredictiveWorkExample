# PredictiveWorkExample

### Load the Libraries
import pandas as pd<br>
import numpy as np<br>
import os<br>
import re<br>
<br>
import torch<br>
import torch.nn.functional as F<br>
import matplotlib.pyplot as plt<br>
from sklearn.preprocessing import MinMaxScaler<br>
import seaborn as sns<br>
<br>
from tqdm.notebook import tqdm_notebook<br>
tqdm_notebook.pandas()<br>
<br>
### Load the feature data file<br>
<br>
# Load file with user summaries<br>
filename = os.path.join(os.path.dirname(__name__), "DataFiles\\feature_dataset.tsv")<br>
df = pd.read_csv(filename,delimiter='\t',low_memory=False)<br>
df.head(2)<br>
<br>
### Select the labelled data<br>
<br>
# read the summary data<br>
filename = os.path.join(os.path.dirname(__name__), "DataFiles\\target_labels.csv")<br>
target = pd.read_csv(filename,delimiter=',',low_memory=False)<br>
<br>
labelled_movie_list = list(target.film_id.unique())<br>
<br>
#### Create unlabelled dataset<br>
<br>
unlabelled_df = df[~df['film_id'].isin(labelled_movie_list)]<br>
<br>
#### Merge feature data with target labels<br>
<br>
# individual summaries<br>
single_summary_df = df[df['film_id'].isin(labelled_movie_list)]<br>
single_df = pd.merge(single_summary_df, target, how="left", on=["film_id"])<br>
<br>
#### Pre-process dataset<br>
lowercase the target labels and strip any extra spaces<br>
<br>
single_df['manual_label'] = single_df['manual_label'].progress_apply(la
X = single_df.drop(['manual_label','summary_id', 'summary','film_id',], axis=1)<br>
y = single_df['manual_label']mbda x: (x.lower().strip()))<br>
<br>
single_df.manual_label.value_counts()<br>
<br>
### Data split into train and test<br>
<br>
from sklearn.model_selection import train_test_split<br>
<br>
# Splitting the labeled data<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)<br>
<br>
print(f"X_train.shape: {X_train.shape}")<br>
print(f"X_test.shape: {X_test.shape}")<br>
<br>
### Feature creation<br>
Convert string embeddings into tensor normalized embeddings<br>
<br>
# convert strings into normalized embeddings<br>
X_train['st_embeddings'] = X_train.st_embeddings.progress_apply(lambda x: F.normalize(eval("torch." + x), p=2, dim=1))<br>
X_test['st_embeddings'] = X_test.st_embeddings.progress_apply(lambda x: F.normalize(eval("torch." + x), p=2, dim=1))<br>
X_train.info()<br>
<br>
#  Normalize sentiment score between [0,1]<br>
<br>
scaler = MinMaxScaler(feature_range=(0, 1))<br>
sentiment_train_scaled = scaler.fit_transform(X_train['sentiment_score'].to_numpy().reshape(-1, 1))<br>
sentiment_test_scaled = scaler.fit_transform(X_test['sentiment_score'].to_numpy().reshape(-1, 1))<br>
<br>
# stack embeddings as np.array<br>
X_train_embeddings = np.vstack(X_train['st_embeddings'].values)<br>
X_test_embeddings = np.vstack(X_test['st_embeddings'].values)<br>
X_train_embeddings.shape<br>
<br>
#### Create a feature set with embeddings and sentiment scores<br>
<br>
X_train_features = np.column_stack((X_train_embeddings, sentiment_train_scaled))<br>
X_test_features = np.column_stack((X_test_embeddings, sentiment_test_scaled))<br>
<br>
### Correct imbalance using class weights<br>
<br>
from sklearn.utils.class_weight import compute_class_weight<br>
<br>
class_weights = compute_class_weight('balanced', classes = np.unique(y), y = y)<br>
class_weights<br>
<br>
class_weight_dict = dict(zip(np.unique(y), class_weights))<br>
class_weight_dict<br>
<br>
### Base-line accuracy<br>
<br>
from sklearn.dummy import DummyClassifier<br>
from sklearn.metrics import accuracy_score<br>
<br>
# Features = embeddings + sentiment scores<br>
dc1 = DummyClassifier(strategy = 'most_frequent')<br>
dc1.fit(X_train_features, y_train)<br>
<br>
acc1 = accuracy_score(y_train, dc1.predict(X_train_features))<br>
<br>
print(f'Baseline accuracy = {acc1:.3f}')<br>
<br>
### Apply SVM <br>
from sklearn.svm import SVC<br>
from sklearn.decomposition import PCA<br>
from sklearn.preprocessing import StandardScaler<br>
from sklearn.pipeline import make_pipeline<br>
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold<br>
<br>
### SVC models hyper parameters defined<br>
<br>
param_C_values = [1, 10, 100, 1000]<br>
param_gamma_values = [0.1, 0.01, 0.001, 0.0001]<br>
<br>
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
<br>
### SVC models with features - embeddings + sentiment score <br>
<br>
from sklearn.model_selection import cross_val_score<br>
<br>
inner_cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)<br>
outer_cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)<br>
<br>
# inner loop for parameter search<br>
gs = GridSearchCV(estimator = SVC(),<br>
                  param_grid = param_grid,<br>
                  cv = inner_cv_folds,<br>
                  scoring = 'accuracy',<br>
                  verbose = 0)<br>
<br>
# outer loop for accuracy scoring<br>
scores = cross_val_score(gs, X = X_train_features, y = y_train, cv = outer_cv_folds, verbose = 99)<br>
<br>
from sklearn.model_selection import learning_curve<br>
<br>
train_sizes, train_scores, test_scores = learning_curve(<br>
    gs, X=X_train_features, y=y_train, cv=outer_cv_folds, <br>
    scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), verbose=0<br>
)<br>
<br>
# Calculate mean and standard deviation for plotting<br>
train_scores_mean = np.mean(train_scores, axis=1)<br>
train_scores_std = np.std(train_scores, axis=1)<br>
test_scores_mean = np.mean(test_scores, axis=1)<br>
test_scores_std = np.std(test_scores, axis=1)<br>
<br>
# Plotting the learning curve<br>
plt.figure(figsize=(10, 6))<br>
plt.title("Learning Curve (SVC)")<br>
plt.xlabel("Training examples")<br>
plt.ylabel("Score")<br>
plt.grid()<br>
<br>
# Plot training scores<br>
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,<br>
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")<br>
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")<br>
<br>
# Plot cross-validation scores<br>
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,<br>
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")<br>
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")<br>
<br>
plt.legend(loc="best")<br>
plt.show()<br>
<br>
<br>
sns.boxplot(x = scores)<br>
sns.swarmplot(x = scores, color = 'black')<br>
plt.title('Model accuracy values (repeated $k$-fold cross validation)')<br>
plt.show()<br>
<br>
<br>
print(f'Mean score using nested cross-validation: {scores.mean():.3f} +/- {scores.std():.3f}')<br>
<br>
cv_folds = StratifiedKFold(n_splits = 5, shuffle = True)<br>
<br>
gs = GridSearchCV(estimator = SVC(),<br>
                  param_grid = param_grid,<br>
                  cv = cv_folds,<br>
                  scoring = 'accuracy',<br>
                  return_train_score = True,<br>
                  refit = True,<br>
                  verbose = 3)<br>
<br>
#### Running model selection on the training set:<br>
<br>
gs.fit(X_train_features, y_train)<br>
<br>
print(gs.best_score_)<br>
best_model_params = gs.best_params_<br>
print(best_model_params)<br>
<br>
best_model = SVC()<br>
best_model.set_params(**best_model_params)<br>
<br>
best_model.probability = True    <br>
best_model.fit(X_train_features, y_train)<br>
<br>
#### Evaluating the best model<br>
<br>
<br>
y_pred = best_model.predict(X_test_features)<br>
<br>
from sklearn.metrics import classification_report, ConfusionMatrixDisplay<br>
print(classification_report(y_test, best_model.predict(X_test_features)))<br>
<br>
