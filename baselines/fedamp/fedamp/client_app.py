"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedamp.dataset import load_data
from fedamp.model import test as test_fn, train as train_fn, Net
from fedamp.utils import combine_aggregated_arrays

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    
    model = Net()

    # for FedAMP: we use 2 models, one that will stay local (model) and the one from aggregated neighbors (aggregated_model) 
    aggregated_model = Net()

    arrays = msg.content.array_records["arrays"]

    # for FedAMP: we also recieve the aggregated weights and coef_self; the aggregatedrecords_key is `mu`
    partial_aggregated_arrays: ArrayRecord = msg.content.array_records['mu']
    coef_self: float = msg.content.array_records['coef_self'] 

    aggregated_arrays: ArrayRecord = combine_aggregated_arrays(partial_aggregated_arrays, arrays, coef_self)
    aggregated_model.load_state_dict(aggregated_arrays.to_torch_state_dict())
    model.load_state_dict(arrays.to_torch_state_dict())
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, _ = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Call the training function: use aggregated_model for the regularization term.
    train_loss = train_fn(
        model,
        aggregated_model,
        trainloader,
        local_epochs,
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    model = Net()
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
