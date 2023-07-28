import argparse

import numpy as np
import pandas as pd 
import wandb
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.linear_model import CoxPHSurvivalAnalysis 
from veteran_Dataset import veteran_Dataset
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_squared_error

from adaboost import AdaBoostF
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

LOG_WANDB = True

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=200, type=int)
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
    event_indicator = y['Status']
    Survival = y['Survival']

    y_structured = np.array([(e, t) for e, t in zip(event_indicator, Survival)], dtype=[('event', bool), ('Survival', float)])
    adaboost_coeff = np.array(adaboost_coeff)

    weak_learner = model.get(0)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X.iloc[ids], y_structured[ids])

    survs = weak_learner.predict_survival_function(X)
    Survivals = weak_learner.event_times_[:-1]
    y_pred = weak_learner.predict(X)
    preds = np.asarray([[fn(t) for t in Survivals] for fn in survs])
    c_index_value = concordance_index_censored(y['Status'], y['Survival'], y_pred)
    c_index_value = c_index_value[0]
    brier_score_value = brier_score(y_structured,y_structured , preds , Survivals)
    integrated_brier_score_value =integrated_brier_score(y_structured, y_structured, preds, Survivals) 
    print("Train Concordance Index  :" , c_index_value)
    print()
    if LOG_WANDB:
        wandb.log({"weak_train_Corcondance-index": c_index_value,
                   "weak_train_Brier_score": brier_score_value, 
                   "weak_train_Integriated_Brier_score": integrated_brier_score_value},
                   #"weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc(y_structured,y_structured,preds , Survivals)},
                  commit=False)
    return {'Corcondance-index': c_index_value}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    event_indicator = y['Status']
    Survival = y['Survival']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, Survival)], dtype=[('Status', bool), ('Survival', float)])
    adaboost_coeff = np.array(adaboost_coeff)
    rank = int(str(name).split('_')[1]) - 1
    error = []
    c_index = []

    #for idx, weak_learner in enumerate(model.get_estimators()):
    # Get the unique event times in the dataset
    event_times_all = np.unique(y['Survival'][y['Status'] == True])
    follow_up_time = np.max(y['Survival'])
    event_times = event_times_all[event_times_all < follow_up_time]
    n_times = len(event_times)

    for idx, weak_learner in enumerate(model.get_estimators()):
        pred = weak_learner.predict(X)

        c_index_value = concordance_index_censored(y['Status'] , y['Survival'] ,pred )
        c_index_value = c_index_value[0]
        survs = weak_learner.predict_survival_function(X)
        #print("survs.shape " , survs.shape)
        survs_interp = np.zeros((survs.shape[0], n_times))
        for i in range(len(survs)):
            survs_interp[i,:] = np.interp(event_times,np.arange(len(survs)), survs[i])
        #To remove
        min_time = min(y[y['Status']==1]['Survival'])
        max_time = max(y[y['Status']==1]['Survival'])
        print("idx : ",idx)
        print("min_y" ,min_time )
        print("max_y" ,max_time )
        print("y['Survival']" ,y['Survival'].sort_values(ascending=True))
        print("survs[0].x", survs[0].x)
        #
        #min_surv = min(survs[0].x)
        #max_surv = max(survs[0].x)
        Survivals =  survs[0].x #(y['Survival'].sort_values(ascending=True)).iloc[:-1] #survs[0].x # np.arange(min_surv , max_surv , len(y))
        #print("y" , len(y))
    
        preds = np.asarray([[fn(t) for t in Survivals] for fn in survs])
        weak_learner_index = str(idx)
        preds_name = 'C:/Users/DELL/Downloads/preds_' + weak_learner_index + '.txt'
        y_name = 'C:/Users/DELL/Downloads/y_' + weak_learner_index + '.txt'
        Survivals_name = 'C:/Users/DELL/Downloads/Survivals_' + weak_learner_index + '.txt'
        pd.DataFrame(preds).to_csv(preds_name, index = True)
        pd.DataFrame(y).to_csv(y_name, index = True)
        pd.DataFrame(Survivals).to_csv(Survivals_name, index = True)
        survival_times = np.array([Survivals[np.argmax(surv.y <= 0.5)] for surv in survs])
        integrated_brier_score_value =integrated_brier_score(y_structured, y_structured, survs_interp, event_times) 

        err = np.where(np.logical_or(y['Status'] == 1, np.logical_and(y['Status'] == 0, survival_times < y['Survival'])),
               np.abs(survival_times - y['Survival']),
               0)
        err = (err-np.min(err))/(np.max(err)-np.min(err))

        # Centering
        #mean = np.mean(err)
        #centered_err = err - mean

        # Scaling
        #std = np.std(centered_err)
        #err = centered_err / std
        error.append(np.dot(adaboost_coeff , err))        #Calculate the weighted error over all the samples for the current client                          
        c_index.append(c_index_value)
        print("weak_train_Corcondance-index",c_index_value)
        print("weak_train_Integriated_Brier_score",integrated_brier_score_value)
        if idx == rank:
            if LOG_WANDB: 
                wandb.log({"weak_train_Corcondance-index": c_index_value,
                           "weak_train_Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, Survivals)},
             
                  commit=False)
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    #print("weak_train_Integriated_Brier_score", integrated_brier_score(y_structured, y_structured, preds, Survivals))
    error.append(np.sum(error)) # ou .mean(error) 
    return {'errors': error }, {'Concordance-index': c_index}

    
@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    X, y = val_loader
    y.to_csv('C:/Users/DELL/Downloads/y_Veteran_test.csv', index=True)
    event_indicator = y['Status'].astype(bool)
    Survival = y['Survival']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, Survival)], dtype=[('Status', bool), ('Survival', float)])
    print("Number of estimators ", model.n_estimators_)
    print("Estimator Weights " , model.estimator_weights_)
    y_pred = model.predict(X)
    print("Shape of final prediction: ", y_pred.shape)
    print("Final prediction : ",  y_pred)
    #print("Survival Function prediction :" , model.predict_surv_function(X))
    c_index_value = concordance_index_censored(y['Status'], y['Survival'], y_pred)
    c_index_value = c_index_value[0]
    for idx,weak_learner in enumerate(model.get_estimators()):
        survs_weak_learner = weak_learner.predict_survival_function(X)
        print("len(survs_weak_learner)" , len(survs_weak_learner))
    #survs = model.predict_surv_function(X)
    

    min_time = min(y['Survival'])
    max_time = max(y['Survival'])
    Survivals = np.arange(min_time , max_time , len(y))
    #preds = np.asarray([[fn(t) for t in Survivals] for fn in survs])
    #print("Integriated_Brier_score", integrated_brier_score(y_structured, y_structured, preds, Survivals))
    if LOG_WANDB:
         wandb.log({"Model_Corcondance-index": c_index_value},
                    #"Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, Survivals)},
                  commit=False)

    return {'Concordance-index': c_index_value}

wandb.init(project="Federated_Learning_Veteran",settings=wandb.Settings(_service_wait=120))
federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False) #Changed from to 50054
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_Veteran",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=CoxPHSurvivalAnalysis()),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = veteran_Dataset()


fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
