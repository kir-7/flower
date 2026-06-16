"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

from collections.abc import Callable, Iterable
from logging import INFO, WARNING

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common import log

from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes
    

import copy
from typing import Dict, Set, Tuple

class FedAMP(FedAvg):
    
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        aggregatedrecord_key: str = "mu",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
    ) -> None:
        
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn
        )

        self.aggregatedrecord_key = aggregatedrecord_key
        
        # store the initial arrays
        self.initial_arrays: ArrayRecord|None = None

        # set of all client arrays
        self.client_arrays: Dict[str, ArrayRecord] = {}

        # set of all client ids 
        self.client_ids: Set = set()

        # set of prev round clients
        self.prev_round_clients: Set = set()


    def _construct_messages(
        self, records: Dict[str, RecordDict], node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the client specific RecordDict payload."""
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=records[node_id],
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round            

        # for all these node_ids get their latest model, if not exist then replace with the initial model
        curr_round_client_arrays: Dict[str, ArrayRecord] = {}
        for node_id in node_ids:
            if node_id in self.client_ids: curr_round_client_arrays[node_id] = self.client_arrays[node_id]
            else: curr_round_client_arrays[node_id] = self.initial_arrays  # we will sent this in self.client_weights in aggregate_fn

        # for FedAMP: other than personalized client model, we want to send the partial aggregated model `mu` 
        # and the self weight (coef_self) which will be used to aggregate on client side
        partial_aggregated_arrays, coef_self = self._compute_aggregated_arrays(curr_round_client_arrays)  # this uses the curr_round_client_arrays, self.prev_round_clients to calculate sim weights

        # share the coef_self: Float via config
        config['coef_self'] = coef_self

        # Construct messages
        records: Dict[str, RecordDict] = {node_id: RecordDict({
            self.arrayrecord_key: curr_round_client_arrays[node_id], 
            self.aggregatedrecord_key:partial_aggregated_arrays[node_id], 
            self.configrecord_key: config                                                               
        }) for node_id in node_ids} 
        
        return self._construct_messages(records, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            #  for FedAMP: no need to aggregate any client arrays, instead update the self.client_arrays and 
            #              self.client_ids and self.prev_round_clients
            
            # clear prev round clients and store the clients for this round
            self.prev_round_clients: Set = set()

            for msg in replies:
                self.client_arrays[msg.metadata.src_node_id] = msg.content.array_records[self.arrayrecord_key] 
                if msg.metadata.src_node_id not in self.client_ids: self.client_ids.add(msg.metadata.src_node_id)
                self.prev_round_clients.add(msg.metadata.src_node_id)            

            # Aggregate MetricRecords as usual
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        records: Dict[str, RecordDict] = {node_id: RecordDict({
            self.arrayrecord_key: self.client_arrays.get(node_id, self.initial_arrays), 
            self.configrecord_key: config                                                               
        }) for node_id in node_ids} 

        return self._construct_messages(records, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics
    
    def _compute_aggregated_arrays(self, curr_round_client_arrays: Dict[str, ArrayRecord]) -> Tuple[Dict[str, ArrayRecord], float]:
        # use the curr_round_model_arrays, prev_round_clients to calculate the sims
        # return the partial_aggregated_arrays, and coef_sim
        raise NotImplementedError("Not yet implemented")

    def start(
            self, 
            grid: Grid, 
            initial_arrays: ArrayRecord, 
            num_rounds: int = 3, 
            timeout: float = 3600, 
            train_config: ConfigRecord | None = None, 
            evaluate_config: ConfigRecord | None  = None, 
            evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None):

        # start fn provided in base Startegy class, use this to store the initial_arrays
        self.initial_arrays = copy.deepcopy(initial_arrays) # create a copy incase later initial_arrays gets changed inplace.   
        return super().start(grid, initial_arrays, num_rounds, timeout, train_config, evaluate_config, evaluate_fn)