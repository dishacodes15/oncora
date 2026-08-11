"""
server.py
Flower federated learning server.
Coordinates training rounds across all hospital clients using FedProx -
averages the weights received from each hospital into a single global
model, same as FedAvg, but each hospital's LOCAL training is regularized
toward the global model via a proximal term (see client.py's fit()).
This is intended to help with client heterogeneity (hospitals with
different data distributions/amounts diverging too far from the global
model during local training) compared to plain FedAvg.
No raw data ever passes through here, only model weights.
"""

import flwr as fl
from flwr.server.strategy import FedProx
import json
from datetime import datetime

# ---- Config ----
NUM_ROUNDS = 5          # how many federated rounds to run
MIN_CLIENTS = 3         # wait until all 3 hospitals are connected before starting
MIN_FIT_CLIENTS = 3     # all 3 must participate in each training round
MIN_EVAL_CLIENTS = 3    # all 3 must participate in each evaluation round

# FedProx proximal term coefficient (mu). 0.0 makes FedProx mathematically
# identical to FedAvg - this is the one new hyperparameter FedProx adds.
# 0.01 is a modest starting value; the original FedProx paper uses higher
# values (up to 1.0) for more heterogeneous/non-convex settings. This is
# a tunable knob, not a validated clinical constant - worth stating as
# such if asked in viva.
PROXIMAL_MU = 0.01

# Track accuracy per round for comparison graphs later
round_history = []


def fit_config(server_round: int):
    """Send round number to clients so they can log it."""
    return {"server_round": server_round}


def evaluate_config(server_round: int):
    return {"server_round": server_round}


def weighted_average(metrics):
    """
    FedAvg aggregation for evaluation metrics.
    Each hospital's accuracy is weighted by how many images it has -
    hospitals with more data have more influence on the global metric.
    This is fairer than a simple average when hospitals have different dataset sizes.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = sum(accuracies) / total_examples

    # Log for later use in comparison graphs
    round_history.append({"accuracy": avg_accuracy, "timestamp": str(datetime.now())})
    print(f"\nGlobal Model Accuracy this round: {avg_accuracy:.4f}\n")

    return {"accuracy": avg_accuracy}


strategy = FedProx(
    fraction_fit=1.0,               # use 100% of available clients each round
    fraction_evaluate=1.0,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVAL_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    proximal_mu=PROXIMAL_MU,
    # NOTE: FedProx automatically merges "proximal_mu" into the config
    # dict sent to every client's fit() call, on top of whatever
    # on_fit_config_fn (fit_config, above) already returns - no change
    # needed to fit_config() itself. Confirmed against the installed
    # flwr package's FedProx.configure_fit() source.
)


def save_history():
    """Save round-by-round accuracy for plotting in the dashboard."""
    with open("../docs/fl_round_history.json", "w") as f:
        json.dump(round_history, f, indent=2)
    print(f"Round history saved to docs/fl_round_history.json")


if __name__ == "__main__":
    print(f"Starting Flower server (FedProx, mu={PROXIMAL_MU}) - waiting for {MIN_CLIENTS} hospital clients...")
    print(f"Running {NUM_ROUNDS} federated rounds\n")

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    save_history()

    print("\n--- Federated Learning Complete ---")
    print(f"Rounds completed: {NUM_ROUNDS}")
    if history.metrics_distributed:
        final_acc = history.metrics_distributed.get("accuracy", [])
        if final_acc:
            print(f"Final global accuracy: {final_acc[-1][1]:.4f}")
