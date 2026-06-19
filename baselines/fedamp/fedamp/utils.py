"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

from flwr.common import ArrayRecord, Array
from flwr.app import Context

from fedamp.strategy import FedAMP
from flwr.serverapp.strategy.fedavg import FedAvg
from flwr.serverapp.strategy.fedprox import FedProx


import matplotlib.pyplot as plt

def combine_aggregated_arrays(
        aggregated_arrays: ArrayRecord, 
        old_arrays: ArrayRecord, 
        coef_self: float) -> ArrayRecord:        
    result_np = {}
    for key, val in aggregated_arrays.items():
        result_np[key] = (old_arrays[key].numpy() * coef_self) + val.numpy()
    return ArrayRecord({k: Array(v) for k, v in result_np.items()})

def get_startegy(context: Context):

    fraction_train: float = float(context.run_config["fraction-train"])
    algorithm = context.run_config['algorithm']
    if algorithm == 'fedamp':
        alphaK = float(context.run_config['alphaK'])
        sigma = float(context.run_config['sigma'])
    
        # Initialize FedAMP strategy
        strategy = FedAMP(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=2,
            alphaK=alphaK,
            sigma=sigma
        )
    elif algorithm == 'fedavg':
        strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
        )

    elif algorithm == 'fedprox':
        strategy = FedProx(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            proximal_mu=context.run_config['fedprox_mu']
        )
    else:
        raise ValueError(f"algortithm: {algorithm } not supported!")

    return strategy


def plot_history(history: dict, save_fig_path: str = "fedamp_plot.png"):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(history["train_rounds"], history["train_losses"], 
                label="Train Loss", marker="o", color="blue")
    axs[0].plot(history["eval_rounds"], history["eval_losses"], 
                label="Eval Loss", marker="s", color="orange")
    axs[0].set_title("Loss over Server Rounds")
    axs[0].set_xlabel("Server Round")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.7)

    axs[1].plot(history["eval_rounds"], history["eval_accs"], 
                label="Eval Accuracy", marker="s", color="green")
    axs[1].set_title("Accuracy over Server Rounds")
    axs[1].set_xlabel("Server Round")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"Plot successfully saved to {save_fig_path}")
        
    plt.show()


def compare_histories(histories, configuration="", save_fig_path="comparison.png"):

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for history in histories:
        label = history["label"]
        axs[0].plot(history["train_rounds"],history["train_losses"],marker="o",label=label)
        axs[1].plot(history["eval_rounds"],history["eval_losses"],marker="s",label=label)
        axs[2].plot(history["eval_rounds"],history["eval_accs"],marker="^",label=label)

    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Server Round")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend()

    axs[1].set_title("Eval Loss")
    axs[1].set_xlabel("Server Round")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].legend()

    axs[2].set_title("Eval Accuracy")
    axs[2].set_xlabel("Server Round")
    axs[2].set_ylabel("Accuracy")
    axs[2].grid(True, linestyle="--", alpha=0.7)
    axs[2].legend()

    plt.suptitle(configuration, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_fig_path}")

    plt.show()