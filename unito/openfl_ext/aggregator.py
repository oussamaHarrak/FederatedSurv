from logging import getLogger
import queue

from openfl.component.aggregation_functions import WeightedAverage
from openfl.component.aggregator import *
from openfl.protocols import ModelProto
from openfl.utilities.logs import write_metric
from openfl.utilities import TaskResultKey
from openfl.utilities import TensorKey

from .tensor_codec import TensorCodec
from .protocols import utils
from .tensor_db import TensorDB
from .identity import Identity
from .aggregate_random_forest import AggregateRandomForest


class Aggregator(Aggregator):
    r"""An Aggregator is the central node in federated learning.

    Args:
        aggregator_uuid (str): Aggregation ID.
        federation_uuid (str): Federation ID.
        authorized_cols (list of str): The list of IDs of enrolled collaborators.
        init_state_path* (str): The location of the initial weight file.
        last_state_path* (str): The file location to store the latest weight.
        best_state_path* (str): The file location to store the weight of the best model.
        db_store_rounds* (int): Rounds to store in TensorDB.
        nn (bool): True if the model is a neural network, False otherwise

    Note:
        \* - plan setting.
    """

    def __init__(self,

                 aggregator_uuid,
                 federation_uuid,
                 authorized_cols,

                 init_state_path,
                 best_state_path,
                 last_state_path,

                 assigner,

                 rounds_to_train=256,
                 log_metric_callback=None,
                 single_col_cert_common_name=None,
                 compression_pipeline=None,
                 db_store_rounds=1,

                 nn=False,

                 **kwargs):
        """Initialize."""
        self.round_number = 0
        self.single_col_cert_common_name = single_col_cert_common_name

        self.nn = nn

        if self.single_col_cert_common_name is not None:
            self._log_big_warning()
        else:
            # FIXME: '' instead of None is just for protobuf compatibility.
            # Cleaner solution?
            self.single_col_cert_common_name = ''

        self.rounds_to_train = rounds_to_train

        # if the collaborator requests a delta, this value is set to true
        self.authorized_cols = authorized_cols
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.assigner = assigner
        self.quit_job_sent_to = []

        self.tensor_db = TensorDB(self.nn)
        # FIXME: I think next line generates an error on the second round
        # if it is set to 1 for the aggregator.
        self.db_store_rounds = db_store_rounds
        self.compression_pipeline = compression_pipeline
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.logger = getLogger(__name__)

        self.init_state_path = init_state_path
        self.best_state_path = best_state_path
        self.last_state_path = last_state_path

        self.best_tensor_dict: dict = {}
        self.last_tensor_dict: dict = {}

        self.metric_queue = queue.Queue()
        self.best_model_score = None

        if self.nn:
            if kwargs.get('initial_tensor_dict', None) is not None:
                self._load_initial_tensors_from_dict(kwargs['initial_tensor_dict'])
                self.model = utils.construct_model_proto(
                    tensor_dict=kwargs['initial_tensor_dict'],
                    round_number=0,
                    tensor_pipe=self.compression_pipeline)
            else:
                self.model: ModelProto = utils.load_proto(self.init_state_path)
                self._load_initial_tensors()  # keys are TensorKeys
        else:
            self.model = utils.construct_model_proto(
                tensor_dict={'model': None},
                round_number=0,
                tensor_pipe=self.compression_pipeline)

        self.log_dir = f'logs/{self.uuid}_{self.federation_uuid}'

        self.collaborator_tensor_results = {}  # {TensorKey: nparray}}

        # these enable getting all tensors for a task

        self.collaborator_tasks_results = {}  # {TaskResultKey: list of TensorKeys}

        self.collaborator_task_weight = {}  # {TaskResultKey: data_size}

        self.log_metric = write_metric

    def send_local_task_results(self, collaborator_name, round_number, task_name,
                                data_size, named_tensors):
        """
        RPC called by collaborator.

        Transmits collaborator's task results to the aggregator.

        Args:
            collaborator_name: str
            task_name: str
            round_number: int
            data_size: int
            named_tensors: protobuf NamedTensor
        Returns:
             None
        """
        self.logger.info(
            f'Collaborator {collaborator_name} is sending task results '
            f'for {task_name}, round {round_number}'
        )

        task_key = TaskResultKey(task_name, collaborator_name, round_number)

        # we mustn't have results already
        if self._collaborator_task_completed(
                collaborator_name, task_name, round_number
        ):
            raise ValueError(
                f'Aggregator already has task results from collaborator {collaborator_name}'
                f' for task {task_key}'
            )

        # initialize the list of tensors that go with this task
        # Setting these incrementally is leading to missing values
        task_results = []

        # go through the tensors and add them to the tensor dictionary and the
        # task dictionary
        for named_tensor in named_tensors:
            # sanity check that this tensor has been updated
            if named_tensor.round_number != round_number:
                raise ValueError(
                    f'Collaborator {collaborator_name} is reporting results for the wrong round.'
                    f' Exiting...'
                )

            # quite a bit happens in here, including decompression, delta
            # handling, etc...
            tensor_key, nparray = self._process_named_tensor(
                named_tensor, collaborator_name
            )
            if 'metric' in tensor_key.tags:
                metric_dict = {
                    'metric_origin': tensor_key.tags[-1],
                    'task_name': task_name,
                    'metric_name': tensor_key.tensor_name,
                    'metric_value': nparray,
                    'round': round_number}
                self.log_metric(tensor_key.tags[-1], task_name,
                                tensor_key.tensor_name, nparray, round_number)
                self.logger.metric(f'Round {round_number}, collaborator {tensor_key.tags[-1]} '
                                   f'{task_name} result {tensor_key.tensor_name}:\t{nparray}')
                self.metric_queue.put(metric_dict)

            task_results.append(tensor_key)
            # By giving task_key it's own weight, we can support different
            # training/validation weights
            # As well as eventually supporting weights that change by round
            # (if more data is added)
            self.collaborator_task_weight[task_key] = data_size

        self.collaborator_tasks_results[task_key] = task_results

        self._end_of_task_check(task_name)

    def _process_named_tensor(self, named_tensor, collaborator_name):
        """
        Extract the named tensor fields.

        Performs decompression, delta computation, and inserts results into TensorDB.

        Args:
            named_tensor:       NamedTensor (protobuf)
                protobuf that will be extracted from and processed
            collaborator_name:  str
                Collaborator name is needed for proper tagging of resulting
                tensorkeys

        Returns:
            tensor_key : TensorKey (named_tuple)
                The tensorkey extracted from the protobuf
            nparray : np.array
                The numpy array associated with the returned tensorkey
        """
        raw_bytes = named_tensor.data_bytes
        metadata = [{'int_to_float': proto.int_to_float,
                     'int_list': proto.int_list,
                     'bool_list': proto.bool_list,
                     'model': proto.model}
                    for proto in named_tensor.transformer_metadata]
        # The tensor has already been transfered to aggregator,
        # so the newly constructed tensor should have the aggregator origin
        tensor_key = TensorKey(
            named_tensor.name,
            self.uuid,
            named_tensor.round_number,
            named_tensor.report,
            tuple(named_tensor.tags)
        )
        tensor_name, origin, round_number, report, tags = tensor_key
        assert ('compressed' in tags or 'lossy_compressed' in tags), (
            f'Named tensor {tensor_key} is not compressed'
        )
        if 'compressed' in tags:
            dec_tk, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=True
            )
            dec_name, dec_origin, dec_round_num, dec_report, dec_tags = dec_tk
            # Need to add the collaborator tag to the resulting tensor
            if type(dec_tags) == str:
                new_tags = tuple([dec_tags] + [collaborator_name])
            else:
                new_tags = tuple(list(dec_tags) + [collaborator_name])
            # layer.agg.n.trained.delta.col_i
            decompressed_tensor_key = TensorKey(
                dec_name, dec_origin, dec_round_num, dec_report, new_tags
            )
        if 'lossy_compressed' in tags:
            dec_tk, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=False
            )
            dec_name, dec_origin, dec_round_num, dec_report, dec_tags = dec_tk
            if type(dec_tags) == str:
                new_tags = tuple([dec_tags] + [collaborator_name])
            else:
                new_tags = tuple(list(dec_tags) + [collaborator_name])
            # layer.agg.n.trained.delta.lossy_decompressed.col_i
            decompressed_tensor_key = TensorKey(
                dec_name, dec_origin, dec_round_num, dec_report, new_tags
            )

        if 'delta' in tags:
            base_model_tensor_key = TensorKey(
                tensor_name, origin, round_number, report, ('model',)
            )
            base_model_nparray = self.tensor_db.get_tensor_from_cache(
                base_model_tensor_key
            )
            if base_model_nparray is None:
                raise ValueError(f'Base model {base_model_tensor_key} not present in TensorDB')
            final_tensor_key, final_nparray = self.tensor_codec.apply_delta(
                decompressed_tensor_key,
                decompressed_nparray, base_model_nparray
            )
        else:
            final_tensor_key = decompressed_tensor_key
            final_nparray = decompressed_nparray

        assert (final_nparray is not None), f'Could not create tensorkey {final_tensor_key}'
        self.tensor_db.cache_tensor({final_tensor_key: final_nparray})
        self.logger.debug(f'Created TensorKey: {final_tensor_key}')

        return final_tensor_key, final_nparray

    def _end_of_task_check(self, task_name):
        """
        Check whether all collaborators who are supposed to perform the task complete.

        Args:
            task_name : str
                The task name to check

        Returns:
            complete : boolean
                Is the task done
        """
        if self._is_task_done(task_name):
            # now check for the end of the round
            self._end_of_round_check()

    def _end_of_round_check(self):
        """
        Check if the round complete.

        If so, perform many end of round operations,
        such as model aggregation, metric reporting, delta generation (+
        associated tensorkey labeling), and save the model

        Returns:
            None
        """
        if not self._is_round_done():
            return

        # Compute all validation related metrics
        all_tasks = self.assigner.get_all_tasks_for_round(self.round_number)
        for task_name in all_tasks:
            self._compute_validation_related_task_metrics(task_name)

        # Once all of the task results have been processed
        # Increment the round number
        self.round_number += 1

        # Save the latest model
        self.logger.info(f'Saving round {self.round_number} model...')
        self._save_model(self.round_number, self.last_state_path)

        # TODO This needs to be fixed!
        if self._time_to_quit():
            self.logger.info('Experiment Completed. Cleaning up...')
        else:
            self.logger.info(f'Starting round {self.round_number}...')

        # Cleaning tensor db
        self.tensor_db.clean_up(self.db_store_rounds)

    def _compute_validation_related_task_metrics(self, task_name):
        """
        Compute all validation related metrics.

        Args:
            task_name : str
                The task name to compute
        """
        # By default, print out all of the metrics that the validation
        # task sent
        # This handles getting the subset of collaborators that may be
        # part of the validation task
        collaborators_for_task = self.assigner.get_collaborators_for_task(
            task_name, self.round_number)
        # The collaborator data sizes for that task
        collaborator_weights_unnormalized = {
            c: self.collaborator_task_weight[TaskResultKey(task_name, c, self.round_number)]
            for c in collaborators_for_task}
        weight_total = sum(collaborator_weights_unnormalized.values())
        collaborator_weight_dict = {
            k: v / weight_total
            for k, v in collaborator_weights_unnormalized.items()
        }

        # The validation task should have just a couple tensors (i.e.
        # metrics) associated with it. Because each collaborator should
        # have sent the same tensor list, we can use the first
        # collaborator in our subset, and apply the correct
        # transformations to the tensorkey to resolve the aggregated
        # tensor for that round
        task_agg_function = self.assigner.get_aggregation_type_for_task(task_name)
        task_key = TaskResultKey(task_name, collaborators_for_task[0], self.round_number)
        for tensor_key in self.collaborator_tasks_results[task_key]:
            tensor_name, origin, round_number, report, tags = tensor_key
            assert (tags[-1] == collaborators_for_task[0]), (
                f'Tensor {tensor_key} in task {task_name} has not been processed correctly'
            )

            # Strip the collaborator label, and lookup aggregated tensor
            new_tags = tuple(tags[:-1])
            agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
            agg_tensor_name, agg_origin, agg_round_number, agg_report, agg_tags = agg_tensor_key
            agg_function = WeightedAverage() if 'metric' in tags else (
                task_agg_function if self.nn else AggregateRandomForest())
            agg_results = self.tensor_db.get_aggregated_tensor(
                agg_tensor_key, collaborator_weight_dict, aggregation_function=agg_function)
            if report:
                # Print the aggregated metric
                metric_dict = {
                    'metric_origin': 'Aggregator',
                    'task_name': task_name,
                    'metric_name': tensor_key.tensor_name,
                    'metric_value': agg_results,
                    'round': round_number}

                if agg_results is None:
                    self.logger.warning(
                        f'Aggregated metric {agg_tensor_name} could not be collected '
                        f'for round {self.round_number}. Skipping reporting for this round')
                if agg_function:
                    self.logger.metric(f'Round {round_number}, aggregator: {task_name} '
                                       f'{agg_function} {agg_tensor_name}:\t{agg_results:.4f}')
                else:
                    self.logger.metric(f'Round {round_number}, aggregator: {task_name} '
                                       f'{agg_tensor_name}:\t{agg_results:.4f}')
                self.log_metric('Aggregator', task_name, tensor_key.tensor_name,
                                agg_results, round_number)
                self.metric_queue.put(metric_dict)
                # TODO Add all of the logic for saving the model based
                #  on best accuracy, lowest loss, etc.
                if 'validate_agg' in tags:
                    # Compare the accuracy of the model, and
                    # potentially save it
                    if self.best_model_score is None or self.best_model_score < agg_results:
                        self.logger.metric(f'Round {round_number}: saved the best '
                                           f'model with score {agg_results:f}')
                        self.best_model_score = agg_results
                        self._save_model(round_number, self.best_state_path)
            if 'trained' in tags:
                self._prepare_trained(tensor_name, origin, round_number, report, agg_results)

    def _save_model(self, round_number, file_path):
        """
        Save the best or latest model.

        Args:
            round_number: int
                Model round to be saved
            file_path: str
                Either the best model or latest model file path

        Returns:
            None
        """
        # Extract the model from TensorDB and set it to the new model
        og_tensor_dict, _ = utils.deconstruct_model_proto(
            self.model, compression_pipeline=self.compression_pipeline)
        tensor_keys = [
            TensorKey(
                k, self.uuid, round_number, False, ('model',)
            ) for k, v in og_tensor_dict.items()
        ]
        tensor_dict = {}
        for tk in tensor_keys:
            tk_name, _, _, _, _ = tk
            tensor_dict[tk_name] = self.tensor_db.get_tensor_from_cache(tk)
            if tensor_dict[tk_name] is None:
                self.logger.info(f'Cannot save model for round {round_number}. Continuing...')
                return
        if file_path == self.best_state_path:
            self.best_tensor_dict = tensor_dict
        if file_path == self.last_state_path:
            self.last_tensor_dict = tensor_dict
        self.model = utils.construct_model_proto(
            tensor_dict, round_number, self.compression_pipeline)
        utils.dump_proto(self.model, file_path)

    def _prepare_trained(self, tensor_name, origin, round_number, report, agg_results):
        """
        Prepare aggregated tensorkey tags.

        Args:
           tensor_name : str
           origin:
           round_number: int
           report: bool
           agg_results: np.array
        """
        # The aggregated tensorkey tags should have the form of
        # 'trained' or 'trained.lossy_decompressed'
        # They need to be relabeled to 'aggregated' and
        # reinserted. Then delta performed, compressed, etc.
        # then reinserted to TensorDB with 'model' tag

        # First insert the aggregated model layer with the
        # correct tensorkey
        agg_tag_tk = TensorKey(
            tensor_name,
            origin,
            round_number + 1,
            report,
            ('aggregated',)
        )
        self.tensor_db.cache_tensor({agg_tag_tk: agg_results})

        # Create delta and save it in TensorDB
        base_model_tk = TensorKey(
            tensor_name,
            origin,
            round_number,
            report,
            ('model',)
        )
        base_model_nparray = self.tensor_db.get_tensor_from_cache(base_model_tk)
        if self.nn and base_model_nparray is not None:
            delta_tk, delta_nparray = self.tensor_codec.generate_delta(
                agg_tag_tk,
                agg_results,
                base_model_nparray
            )
        else:
            # This condition is possible for base model
            # optimizer states (i.e. Adam/iter:0, SGD, etc.)
            # These values couldn't be present for the base
            # model because no training occurs on the aggregator
            delta_tk, delta_nparray = agg_tag_tk, agg_results

        # Compress lossless/lossy
        compressed_delta_tk, compressed_delta_nparray, metadata = self.tensor_codec.compress(
            delta_tk, delta_nparray
        )

        # TODO extend the TensorDB so that compressed data is
        #  supported. Once that is in place
        # the compressed delta can just be stored here instead
        # of recreating it for every request

        # Decompress lossless/lossy
        decompressed_delta_tk, decompressed_delta_nparray = self.tensor_codec.decompress(
            compressed_delta_tk,
            compressed_delta_nparray,
            metadata
        )

        self.tensor_db.cache_tensor({decompressed_delta_tk: decompressed_delta_nparray})

        # Apply delta (unless delta couldn't be created)
        if self.nn and base_model_nparray is not None:
            self.logger.debug(f'Applying delta for layer {decompressed_delta_tk[0]}')
            new_model_tk, new_model_nparray = self.tensor_codec.apply_delta(
                decompressed_delta_tk,
                decompressed_delta_nparray,
                base_model_nparray
            )
        else:
            new_model_tk, new_model_nparray = decompressed_delta_tk, decompressed_delta_nparray

        # Now that the model has been compressed/decompressed
        # with delta operations,
        # Relabel the tags to 'model'
        (new_model_tensor_name, new_model_origin, new_model_round_number,
         new_model_report, new_model_tags) = new_model_tk
        final_model_tk = TensorKey(
            new_model_tensor_name,
            new_model_origin,
            new_model_round_number,
            new_model_report,
            ('model',)
        )

        # Finally, cache the updated model tensor
        self.tensor_db.cache_tensor({final_model_tk: new_model_nparray})