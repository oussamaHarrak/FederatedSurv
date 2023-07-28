import argparse

import numpy as np
import wandb
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.linear_model import CoxPHSurvivalAnalysis 
from heart_failure_Dataset import heart_failure_Dataset
from sklearn.metrics import mean_squared_error

from adaboost import AdaBoostF
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

LOG_WANDB = True

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=3, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--server", default='localhost', type=str, help="server address")
args = parser.parse_args()

random_state = np.random.RandomState(args.seed)

client_id = 'api'
cert_dir = '../cert'

task_interface = TaskInterface()


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def train_adaboost(model, train_loader, device, optimizer, adaboost_coeff, name):
    X, y = train_loader
    event_indicator = y['DEATH_EVENT'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    adaboost_coeff = np.array(adaboost_coeff)

    weak_learner = model.get(0)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X.iloc[ids], y_structured[ids])
    survs = weak_learner.predict_survival_function(X)
    min_time = min(y['time'])
    max_time = max(y['time'])
    print("time : ", y['time'])
    print("DEATH_EVENT : ", y['DEATH_EVENT'])
    times = np.arange(min_time -2 , max_time + 5)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    metric = weak_learner.score(X,y_structured)
    if LOG_WANDB:
        wandb.log({"weak_train_Corcondance-index": metric,
                   "weak_train_Brier_score": brier_score(y_structured,y_structured , preds , times), 
                   "weak_train_Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, times),
                   "weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc(y_structured,y_structured,preds , times)},
                  commit=False)
    return {'Corcondance-index': metric}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    event_indicator = y['DEATH_EVENT'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    adaboost_coeff = np.array(adaboost_coeff)

    rank = int(str(name).split('_')[1]) - 1
    times = np.arange(0, 290)

    error = []
    c_index = []
    for idx, weak_learner in enumerate(model.get_estimators()):
        survs = weak_learner.predict_survival_function(X)
        preds = np.asarray([[fn(t) for t in times] for fn in survs])
        c_index_value = weak_learner.score(X,y_structured)
        error.append(sum(adaboost_coeff[:idx+1]) * c_index_value)
        c_index.append(c_index_value)

        if idx == rank:
            if LOG_WANDB:
                wandb.log({"weak_train_Corcondance-index": weak_learner.score(X,y_structured),
                   "weak_train_Brier_score": brier_score(y_structured,y_structured,preds,times), 
                   "weak_train_Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, times),
                   "weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc(y_structured,y_structured,preds , times)},
                  commit=False)
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
   
    error.append(sum(adaboost_coeff))

    return {'errors': error}, {'Conrcordance-index': c_index}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    X, y = val_loader
    event_indicator = y['DEATH_EVENT'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    pred = model.predict(np.array(X))
    survs = model.predict_survival_function(X)
    times = np.arange(0, 290)
    preds = np.asarray([[fn(t) for t in times] for fn in survs])
    brier_score = brier_score(y_structured,y_structured,preds,times)
    c_index_value = model.score(X,y_structured)
    if LOG_WANDB:
         wandb.log({"weak_train_Corcondance-index": c_index_value,
                   "weak_train_Brier_score": brier_score, 
                   "weak_train_Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, times),
                   "weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc(y_structured,y_structured,preds , times)},
                  commit=False)

    return {'Concordance-index': c_index_value}

wandb.init(project="Federated_Learning_Heart_Failure",settings=wandb.Settings(_service_wait=120))
federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False) #Changed from to 50054
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_Heart_Failure",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=CoxPHSurvivalAnalysis()),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = heart_failure_Dataset()


fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
