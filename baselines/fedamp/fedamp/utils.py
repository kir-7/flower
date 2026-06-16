"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

from flwr.common import ArrayRecord

def combine_aggregated_arrays(
        aggregated_arrays: ArrayRecord, 
        old_arrays: ArrayRecord, 
        coef_self: float) -> ArrayRecord:
    raise NotImplementedError("Not yet implemented")