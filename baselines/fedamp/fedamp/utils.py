"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

from flwr.common import ArrayRecord, Array

import matplotlib.pyplot as plt

def combine_aggregated_arrays(
        aggregated_arrays: ArrayRecord, 
        old_arrays: ArrayRecord, 
        coef_self: float) -> ArrayRecord:        
    result_np = {}
    for key, val in aggregated_arrays.items():
        result_np[key] = (old_arrays[key].numpy() * coef_self) + val.numpy()
    return ArrayRecord({k: Array(v) for k, v in result_np.items()})


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

