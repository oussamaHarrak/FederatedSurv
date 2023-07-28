import argparse
import sys
import numpy as np
import wandb
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.linear_model import CoxPHSurvivalAnalysis 
from support_feature_Dataset import support_feature_Dataset
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_squared_error
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.integrate import simps

from adaboost import AdaBoostF
import random

from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

LOG_WANDB = True
random.seed(42)

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=100, type=int)
parser.add_argument("--seed", default=1, type=int)
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
    event_indicator = y['event'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    adaboost_coeff = np.array(adaboost_coeff)

    weak_learner = model.get(0)
    ids = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=adaboost_coeff / adaboost_coeff.sum())
    weak_learner.fit(X.iloc[ids], y_structured[ids])
    survs = weak_learner.predict_survival_function(X)
    Survivals = weak_learner.event_times_[:-1]
    preds = np.asarray([[fn(t) for t in Survivals] for fn in survs])
    y_pred = weak_learner.predict(X)
 
    c_index_value = concordance_index_censored(y['event'], y['time'], y_pred)
    c_index_value = c_index_value[0]
    print("Train Min times : " , min(y['time']) )
    print("Train Max times : " , max(y['time']) )
    print("Train Min times_1 :" , min(y[y['event'] == 1]['time']))
    print("Train Max times_1 :" , max(y[y['event'] == 1]['time']))

    """if LOG_WANDB:
        wandb.log({"weak_train_Corcondance-index": c_index_value,
                  "weak_train_Brier_score": brier_score(y_structured,y_structured , preds , Survivals), 
                 "weak_train_Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, Survivals),
                "weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc(y_structured,y_structured,preds , Survivals)},
                  commit=False)"""
  
    return {'Corcondance-index': c_index_value}
 #print("Time events" , weak_learner.event_times_[:-1])
        #preds = np.asarray([[fn(t) for t in times] for fn in survs])
        #brier = brier_score(y_structured, y_structured, preds, times)
        #integriated_brier = integrated_brier_score(y_structured, y_structured, preds, times)
        #cumulative_dynamic_auc0 = cumulative_dynamic_auc(y_structured,y_structured,preds , times)
"""
        surv_function = weak_learner.predict_survival_function(X)
        times = surv_function[0].x 
        squared_diff = (surv_function - surv_function_KM)**2
        err = simps(squared_diff , times)
         min_time = min(y[y['event'] == 1]['time'])
        max_time = max(y['time'])
        times = np.arange(8, 272 )
        print("min_times :" , min(y['time']))
        print("max_times : " ,max(y['time']))
        print("min_times_1 : ", min_time )
        print("max_times_1 : " , max(y[y['event'] == 1]['time']))
"""

@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    event_indicator = y['event'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    adaboost_coeff = np.array(adaboost_coeff)

    rank = int(str(name).split('_')[1]) - 1
    error = []
    c_index = []
   
    for idx, weak_learner in enumerate(model.get_estimators()):
        pred = weak_learner.predict(X)
        c_index_value = concordance_index_censored(y['event'] , y['time'] ,pred )
        c_index_value = c_index_value[0]
        survs = weak_learner.predict_survival_function(X)
       
        time_values = weak_learner.event_times_
        survival_time = np.array([time_values[np.argmax(surv.y <= 0.5)] for surv in survs])

           
     
        #err = np.where(y['event'] == 0, 0, np.abs(survival_times - y['time']))
        err = np.where(np.logical_or(y['event'] == 1, np.logical_and(y['event'] == 0, survival_time < y['time'])),
               np.abs(survival_time - y['time']),
               0)
        #Normalize instead
        err = (err-np.min(err))/(np.max(err)-np.min(err))
        # Centering
        #mean = np.mean(err)
        #centered_err = np.abs(err - mean)

        # Scaling
        #std = np.std(centered_err)
        #err = centered_err / std


         #Possible loss functions : 
                                #L = absolute value of (predicted survival time - actual survival time) (Still needs to be normalized)
                                #L = 1 - c_index
                                #L = brier_score (Problem with the min and the max of time)

         #Calculate the weighted error for each weak learner 1-c_index_value represents the error
                                                              #Li = 1 - c_index_value 

        error.append(np.dot(adaboost_coeff , err))        #Calculate the weighted error over all the samples for the current client                          
        c_index.append(c_index_value)
        if idx == rank:
            if LOG_WANDB: 
                wandb.log({"weak_validate_Corcondance-index": c_index_value },
                           # "weak_brier_scores" : brier }, 
                            #"weak_integriated_brier_score" : integriated_brier,
                            #"weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc0 },
                  commit=False)
           
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    error.append(np.sum(error))# ou .mean(error) 
    
    return {'errors': error }, {'Concordance-index': c_index}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', adaboost_coeff = 'adaboost_coeff', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    
    X, y = val_loader
    y.to_csv('C:/Users/DELL/Downloads/y_test.csv', index=True)

    event_indicator = y['event'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('event', bool), ('time', float)])
    print("Number of estimators ", model.n_estimators_)
    print("Estimator Weights " , model.estimator_weights_)
    #for idx,weak_learner in enumerate(model.get_estimators()) :
     #   surv_function = weak_learner.predict_survival_function(X)
        #print("Weak Learner {idx} : " , surv_function)

    """if np.isnan(model.estimator_weights_).any() :
        
        print("Nan value encountered in the estimator weights , Stopping the experiment...")
        if LOG_WANDB:
            wandb.log({"Model_Corcondance-index": c_index_value},
                  commit=False)
            sys.exit()"""
    y_pred = model.predict(X)
    print("Shape of final prediction: ", y_pred.shape)
    print("Final prediction : ",  y_pred)
    #print("Predicted Survival Function :" ,model.predict_surv_function(X))
    c_index_value = concordance_index_censored(y['event'], y['time'], y_pred)
    c_index_value = c_index_value[0]
    """
    survival_predictions = []
    surv_function_KM = kaplan_meier_estimator(y["event"] , y["time"])
    for idx, weak_learner in enumerate(model.get_estimators()):
        surv_function_weak = weak_learner.predict_survival_function(X)
        survival_predictions.append(surv_function_weak)
    surv_function = np.average(survival_predictions, weights=adaboost_coeff, axis=0)

    squared_diff = (surv_function - surv_function_KM)**2
    err = simps(squared_diff , times)
    """
    if LOG_WANDB:
         wandb.log({"Model_Corcondance-index": c_index_value},
                  commit=False)
  
    return {'Concordance-index': c_index_value}

wandb.init(project="Federated_Learning_support_feature",settings=wandb.Settings(_service_wait=120))
federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False) #Changed from to 50054
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_support_feature1",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=AdaBoostF(base_estimator=CoxPHSurvivalAnalysis(alpha=0.01)),#CoxPHSurvivalAnalysis()),
    optimizer=None,
    framework_plugin='openfl.plugins.frameworks_adapters.generic_adapter.GenericAdapter')
federated_dataset = support_feature_Dataset()


fl_experiment.start(
    model_provider=model_interface,
    task_keeper=task_interface,
    data_loader=federated_dataset,
    rounds_to_train=args.rounds,
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
