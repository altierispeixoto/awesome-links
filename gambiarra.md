# Databricks notebook source
# MAGIC %md # Imports

# COMMAND ----------

import sys, json, logging
import os
import shutil
import ast
import copy
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
import pickle
import lightgbm as lgb

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

try:
  import mlflow
  import mlflow.sklearn
  from hyperopt import fmin, tpe, hp, anneal, Trials
  sys.path.insert(0, '/dbfs/mnt/lib/src/')
  from evaluation import PrecisionRecall, EvalROC
  from utils import load_config, report_performance_metrics, get_reports_fig

except ModuleNotFoundError:
  # Wait libraries install in cluster
  import time
  time.sleep(90)
  import mlflow
  import mlflow.sklearn
  from hyperopt import fmin, tpe, hp, anneal, Trials
  sys.path.insert(0, '/dbfs/mnt/lib/src/')
  from evaluation import PrecisionRecall, EvalROC
  from utils import load_config, report_performance_metrics, get_reports_fig

# COMMAND ----------

# MAGIC %md # Input Config

# COMMAND ----------

dbutils.widgets.text('experiment_config', '')
experiment_config = dbutils.widgets.get('experiment_config')
config = ast.literal_eval(experiment_config)
print(config)

# COMMAND ----------

# MAGIC %md #### Uncomment cell below to run manually

# COMMAND ----------

# experiment_config = """{
#     "database": "dev_churn_weekly",
#     "project_name": "churn_weekly",
#     "config_path": "../config/v9_model_variables",
#     "master_version": "v9",
#     "target": "churn_0G_3W",
#     "balancing": "U50",
#     "train_end_date": "2019-03-31",
#     "train_start_date": "2018-01-01",
#     "n_trials": 5,
#     "space": {
#       # config params
#       "n_date_splits": 5,
#       "trials": "",
#       "search_metric": "logloss",

#       # training param
#       "num_boost_round": 10000,

#       # hyperparameters
#       "objective": "binary",
#       "boosting_type": "gbdt",
#       "metric": ["auc", "binary_logloss"],
#       "num_leaves": {"hp.quniform": ("num_leaves", 3, 70, 1)},
#       "max_depth": {"hp.quniform": ("max_depth", 2, 20, 1)},
#       # "learning_rate": {"hp.loguniform": ("learning_rate", -5, 0)},
#       "learning_rate": {"hp.uniform": ("learning_rate", 0.1, 0.1)},
#       "min_data_in_leaf": {"hp.quniform": ("min_data_in_leaf", 50, 250, 1)},
#       "max_bin": {"hp.quniform": ("max_bin", 100, 256, 1)},
#       "bagging_freq": {"hp.quniform": ("bagging_freq", 1, 6, 1)},
#       "bagging_fraction": {"hp.uniform": ("bagging_fraction", 0.3, 1)},
#       "feature_fraction": {"hp.uniform": ("feature_fraction", 0.3, 1)},
#       "is_unbalance": {"hp.randint": ("is_unbalance", 1)}
#     }
#   }"""

# experiment_config = ast.literal_eval(experiment_config)

# COMMAND ----------

initial_experiment_config = copy.deepcopy(config)
variables_config_path = config['config_path']
project_name = config['project_name']
database = config['database']
master_version = config['master_version']
target = config['target']
balancing = config['balancing']
train_end_date = config['train_end_date']
train_start_date = config['train_start_date']
n_trials = config['n_trials']
space = config['space']
n_date_splits = space['n_date_splits']
variables_config = json.loads(dbutils.notebook.run(variables_config_path, 300))

# COMMAND ----------

# Parse dict to use hp functions correctly
for key, value in space.items():
  if type(value) is dict:
    for space_function, params in value.items():
      space_function = space_function.replace('hp.', '')
      space[key] = getattr(hp, space_function)(*params)

# COMMAND ----------

experiment_name = get_experiment_name(master_version, target, balancing,
                                      n_date_splits, train_end_date)
train_data = get_train_table(master_version, target, balancing)
print(experiment_name)
target = target

# COMMAND ----------

# MAGIC %md ## Load Train Data

# COMMAND ----------

spark.sql('use {}'.format(database))
spark.sql('refresh table {}'.format(train_data))
spark.sql('cache lazy table {}'.format(train_data))
print(train_data)

# Carrega os dados de treinamento
X_train, y_train, df_keys = load_data(train_data, variables_config,
  target, start_date=train_start_date, end_date=train_end_date)

print(X_train.shape)

# COMMAND ----------

X_train.info()

# COMMAND ----------

pd.options.display.max_rows = 999
print(X_train.dtypes)

# COMMAND ----------

# MAGIC %md # Train model

# COMMAND ----------

# Inicia experimento
print('Experiment: ' + experiment_name)
print('Variables Config: ' + variables_config_path)
print('Started: ' + str(datetime.now()))

# COMMAND ----------

experiment_path = get_experiment_files_path(project_name, experiment_name, is_experiment=True)
experiment_tmp_path = experiment_path + 'tmp/'
os.makedirs(os.path.dirname(experiment_tmp_path), exist_ok=True)

# COMMAND ----------

def log_input_params(params, trials):
    # Save variables config artifact
    variables_config_path = experiment_tmp_path + 'variables_config.json'
    with open(variables_config_path, 'w') as outfile:
      json.dump(variables_config, outfile)
    mlflow.log_artifact(variables_config_path)

    # Save experiment config artifact
    experiment_config_path = experiment_tmp_path + 'experiment_config.json'
    with open(experiment_config_path, 'w') as outfile:
      json.dump(initial_experiment_config, outfile)
    mlflow.log_artifact(experiment_config_path)

    # Save params artifact
    params_path = experiment_tmp_path + 'params.json'
    with open(params_path, 'w') as outfile:
      json.dump(params, outfile)
    mlflow.log_artifact(params_path)

    # Log Params
    for key, value in params.items():
      mlflow.log_param(key, value)

    # Log trial iteration
    mlflow.log_param('trial', len(trials.trials))
    print('Starting trial: {}'.format(len(trials.trials)))

    # Log training dates
    mlflow.log_param('train_start_date', train_start_date)
    mlflow.log_param('train_end_date', train_end_date)

# COMMAND ----------

def objective_function(params):
    """Objective function to minimize"""
    metrics_dict = {
      'auc': {
        'function': roc_auc_score,
        'scores_list': []
      },
      'ap': {
        'function': average_precision_score,
        'scores_list': []
      },
      'logloss':{
        'function': log_loss,
        'scores_list': []
      }
    }

    trials = params.pop('trials', None)
    n_date_splits = params.pop('n_date_splits', 5)
    search_metric = params.pop('search_metric', 'logloss')
    num_boost_round = params.pop('num_boost_round', 2000)

    # Cast params from search
    int_variables = [
      'num_leaves', 'max_depth', 'min_data_in_leaf',
      'bagging_freq', 'max_bin'
    ]
    for variable in int_variables:
      params[variable] = int(params[variable])

    if 'is_unbalance' in space:
      params['is_unbalance'] = False if params['is_unbalance'] == 0 else True

    try:
      mlflow.start_run()
    except:
      mlflow.end_run()
      mlflow.start_run()

    print(params)
    log_input_params(params, trials)

    evals_results = []
    dates = df_keys['datareferencia']
    date_split = DateSplit(n_splits=n_date_splits)
    for i, (train_index, test_index) in enumerate(date_split.split(dates)):
        print('Starting fold number: {}'.format(i))
        train_dates = dates.iloc[train_index].unique()
        test_dates = dates.iloc[test_index].unique()

        print('Train dates: from {} to {}'.format(
          train_dates[0].strftime('%Y-%m-%d'), train_dates[-1].strftime('%Y-%m-%d')))
        print('Test dates: from {} to {}'.format(
          test_dates[0].strftime('%Y-%m-%d'), test_dates[-1].strftime('%Y-%m-%d')))

        # Create cv train, valid and test datasets
        X_train_cv, y_train_cv, X_valid, y_valid = \
          split_validation(X_train.iloc[train_index], y_train.iloc[train_index],
                           dates.iloc[train_index])

        lgb_train = lgb.Dataset(X_train_cv, y_train_cv)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        evals_result = {}
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=300,
                        valid_sets=[lgb_train, lgb_valid],
                        evals_result=evals_result,
                        verbose_eval=200)

        evals_results.append(evals_result)
        X_test_cv, y_test_cv = X_train.iloc[test_index], y_train.iloc[test_index]
        y_pred = gbm.predict(X_test_cv)

        # Log cv score metric
        for key, value in metrics_dict.items():
          score = value['function'](y_test_cv, y_pred)
          score = np.round(score, 4)
          value['scores_list'].append(score)
          mlflow.log_metric('cv_{}_fold{}'.format(key, i), score)

        # Generate the cv metrics fig and save
        fig, ax = plt.subplots(1, 2, figsize=(20, 10),
                               gridspec_kw=dict(hspace=0.5, wspace=0.3))
        eval_roc = EvalROC(y_test_cv, y_pred)
        eval_roc.plot(ax=ax[0])
        precision_recall = PrecisionRecall(y_test_cv, y_pred)
        precision_recall.plot(ax=ax[1])
        fig_name = experiment_tmp_path + 'cv_metrics_fold{}.png'.format(i)
        fig.savefig(fig_name)
        mlflow.log_artifact(fig_name)
        plt.close(fig)

    # Generate cv training binary_logloss figs
    fig = get_results_fig(evals_results, metric='binary_logloss')
    fig_name = experiment_tmp_path + 'cv_binary_logloss_figs.png'
    fig.savefig(fig_name)
    mlflow.log_artifact(fig_name)
    plt.close(fig)

    # Generate cv training auc figs
    fig = get_results_fig(evals_results, metric='auc')
    fig_name = experiment_tmp_path + 'cv_auc_figs.png'
    fig.savefig(fig_name)
    mlflow.log_artifact(fig_name)
    plt.close(fig)

    for key, value in metrics_dict.items():
      metric_mean = np.mean(value['scores_list'])
      mlflow.log_metric('mean_{}'.format(key), metric_mean)
      if search_metric == key:
        search_metric_mean = metric_mean

    if search_metric == 'auc' or search_metric == 'ap':
      search_metric_mean = -search_metric_mean

    mlflow.end_run()
    return search_metric_mean

# COMMAND ----------

def run_trials():
  experiment_databricks_path = get_experiment_databricks_path(
    project_name, experiment_name, is_experiment=True)
  experiment_path = get_experiment_path(project_name, experiment_name, is_experiment=True)
  artifact_location = experiment_path + 'runs'

  try:
    mlflow.create_experiment(experiment_databricks_path, artifact_location=artifact_location)
  except:
    pass

  mlflow.set_experiment(experiment_databricks_path)
  trials_path = '{}trials/trials.pkl'.format(experiment_path)
  trials_path = trials_path.replace('dbfs:', '/dbfs')
  trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
  max_trials = 5  # initial max_trials. put something small to not have to wait

  try:  # try to load an already saved trials object, and increase the max
    trials = pickle.load(open(trials_path, 'rb'))
    print("Found saved Trials! Loading...")
    max_trials = len(trials.trials) + trials_step
    print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
  except:  # create a new trials object and start searching
    trials = Trials()

  space['trials'] = trials

  best = fmin(
    fn=objective_function, # function to optimize
    space=space,
    algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
    max_evals=max_trials, # maximum number of iterations
    trials=trials, # logging
    show_progressbar=False
  )

  print('Best: ', best)

  # save the trials object
  os.makedirs(os.path.dirname(trials_path), exist_ok=True)
  with open(trials_path, 'wb') as f:
    pickle.dump(trials, f)

  return best

# COMMAND ----------

for i in range(n_trials):
    best = run_trials()

