import argparse

import numpy as np
import wandb
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.linear_model import CoxPHSurvivalAnalysis 
from heart_failure_Dataset import heart_failure_Dataset
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_squared_error
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.integrate import simps

from adaboost import AdaBoostF
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

LOG_WANDB = True    

parser = argparse.ArgumentParser(description="Script")
parser.add_argument("--rounds", default=50, type=int)
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
    Survivals = weak_learner.event_times_[:-1]
    preds = np.asarray([[fn(t) for t in Survivals] for fn in survs])
    y_pred = weak_learner.predict(X)
 
    c_index_value = concordance_index_censored(y['DEATH_EVENT'], y['time'], y_pred)
    c_index_value = c_index_value[0]
    print("Train Min times : " , min(y['time']) )
    print("Train Max times : " , max(y['time']) )
    print("Train Min times_1 :" , min(y[y['DEATH_EVENT'] == 1]['time']))
    print("Train Max times_1 :" , max(y[y['DEATH_EVENT'] == 1]['time']))

    if LOG_WANDB:
        wandb.log({"weak_train_Corcondance-index": c_index_value,
                  "weak_train_Brier_score": brier_score(y_structured,y_structured , preds , Survivals), 
                 "weak_train_Integriated_Brier_score": integrated_brier_score(y_structured, y_structured, preds, Survivals),
                "weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc(y_structured,y_structured,preds , Survivals)},
                  commit=False)
    return {'Corcondance-index': c_index_value}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device',
                                 adaboost_coeff='adaboost_coeff', name='name', nn=False)
def validate_weak_learners(model, val_loader, device, adaboost_coeff, name):
    X, y = val_loader
    event_indicator = y['DEATH_EVENT'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('DEATH_EVENT', bool), ('time', float)])
    adaboost_coeff = np.array(adaboost_coeff)

    rank = int(str(name).split('_')[1]) - 1

    c_index = []
    error = []
    miss = []
    #times = np.arange(8,280)
    min_time = min(y[y['DEATH_EVENT'] == 1]['time'])
    max_time = max(y['time'])
    times = np.arange(8, 272 )
    print("min_times :" , min(y['time']))
    print("max_times : " ,max(y['time']))
    print("min_times_1 : ", min_time )
    print("max_times_1 : " , max(y[y['DEATH_EVENT'] == 1]['time']))
    #print("times : " , times)
    #surv_function_KM = kaplan_meier_estimator(y["DEATH_EVENT"] , y["time"])
    for idx, weak_learner in enumerate(model.get_estimators()):
        pred = weak_learner.predict(X)
        c_index_value = concordance_index_censored(y['DEATH_EVENT'] , y['time'] ,pred )
        c_index_value = c_index_value[0]
        survs = weak_learner.predict_survival_function(X)
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
        """
        time_values = weak_learner.event_times_
        survival_times = []
        for i in range((survs.shape)[0]):
            survs_sample = survs[i]
            survival_time = time_values[np.argmax(survs_sample.y <= 0.5)] #We estimate the median 
            survival_times.append(survival_time)
        err = abs(survival_times - y['time'])
     

        # Centering
        mean = np.mean(err)
        centered_err = err - mean

        # Scaling
        std = np.std(centered_err)
        err = centered_err / std
        thershold = np.mean(err)
        for i in range(len(err)):
            if ( err[i] > thershold):
                miss.append(i)
        error.append(sum(adaboost_coeff[miss]))

         #Possible loss functions : 
                                #L = absolute value of (predicted survival time - actual survival time) (Still needs to be normalized)
                                #L = 1 - c_index
                                #L = brier_score (Problem with the min and the max of time)

       # error.append(adaboost_coeff[idx]*err) #Calculate the weighted error for each weak learner 1-c_index_value represents the error
                                                              #Li = 1 - c_index_value 

        c_index.append(c_index_value)
        if idx == rank:
            if LOG_WANDB: 
                wandb.log({"weak_validate_Corcondance-index": c_index_value },
                           # "weak_brier_scores" : brier }, 
                            #"weak_integriated_brier_score" : integriated_brier,
                            #"weak_train_cumulative_dynamic_auc": cumulative_dynamic_auc0 },
                  commit=False)
    # TODO: piccolo trick, alla fine di ogni vettore errori viene mandata la norma dei pesi locali
    
    error.append(sum(adaboost_coeff))
    return {'errors': error }, {'misprediction': miss }, {'Concordance-index' , c_index}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', nn=False)
def adaboost_update(model, val_loader, device):
    return {'adaboost_update': 0}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device', adaboost_coeff = 'adaboost_coeff', name='name', nn=False)
def validate_adaboost(model, val_loader, device, name):
    if model is None :
        print("Model not found")
    X, y = val_loader
    event_indicator = y['DEATH_EVENT'].astype(bool)
    time = y['time']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, time)], dtype=[('DEATH_EVENT', bool), ('time', float)])
    print("Number of estimators ", model.n_estimators_)
    print("Estimator Weights " , model.estimator_weights_)
    print("Estimators : " ,model.get_estimators() )
    y_pred = model.predict(X)
    print("Shape of final prediction: ", y_pred.shape)
    print("Final prediction : ",  y_pred)
    c_index_value = concordance_index_censored(y['DEATH_EVENT'], y['time'], y_pred)
    c_index_value = c_index_value[0]
    """
    survival_predictions = []
    surv_function_KM = kaplan_meier_estimator(y["DEATH_EVENT"] , y["time"])
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

wandb.init(project="Federated_Learning_Heart_Failure",settings=wandb.Settings(_service_wait=120))
federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False) #Changed from to 50054
fl_experiment = FLExperiment(federation=federation, experiment_name="AdaboostF_Heart_Failure1",
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
    rounds_to_train=10,#args.rounds
    opt_treatment='CONTINUE_GLOBAL'
)

fl_experiment.stream_metrics(tensorboard_logs=False)
fl_experiment.remove_experiment_data()
