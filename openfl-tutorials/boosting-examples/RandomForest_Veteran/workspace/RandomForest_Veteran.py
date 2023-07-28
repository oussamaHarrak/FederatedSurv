import argparse
import wandb
import numpy as np
from sklearn.exceptions import NotFittedError
from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc
from veteran_Dataset import veteran_Dataset
from sksurv.metrics import concordance_index_censored
from openfl.interface.interactive_api.experiment import FLExperiment, TaskInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation
from random_forest import MyRandomSurvivalForest

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


@task_interface.register_fl_task(model='model', data_loader='train_loader', device='device', optimizer='optimizer' )
def train(model, train_loader, device, optimizer):
    X, y = train_loader
    event_indicator = y['Status']
    Survival = y['Survival']
    y_structured = np.array([(e, t) for e, t in zip(event_indicator, Survival)], dtype=[('Status', bool), ('Survival', float)])
    model.fit(X, y_structured)
    y_pred = model.predict(X)
    c_index = concordance_index_censored(y['Status'], y['Survival'], y_pred)
    c_index = c_index[0]
    if LOG_WANDB:
        wandb.log({"train_Corcondance-index": c_index},commit=False)
    return {'Corcondance index': c_index}


@task_interface.register_fl_task(model='model', data_loader='val_loader', device='device')
def validate(model, val_loader, device):
    try:
        if model is not None:
            X, y = val_loader
            y_pred = model.predict(X)
            c_index = concordance_index_censored(y['Status'], y['Survival'], y_pred)
            c_index = c_index[0]
            if LOG_WANDB:
                wandb.log({"Model validate_Corcondance-index": c_index},commit=False)
        else:
            print("Model not found")
            c_index = 0
    except NotFittedError:
        print("Model is not yet fit")
        c_index = 0

    return {'Concordance index': c_index}

wandb.init(project="Federated_Learning_Veteran2",settings=wandb.Settings(_service_wait=120))

federation = Federation(client_id=client_id, director_node_fqdn=args.server, director_port='50054', tls=False)
fl_experiment = FLExperiment(federation=federation, experiment_name="RandomForest_Veteran",
                             serializer_plugin='openfl.plugins.interface_serializer.dill_serializer.DillSerializer',
                             load_default_plan=False, nn=False)
model_interface = ModelInterface(
    model=MyRandomSurvivalForest(),
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
