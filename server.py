import flwr as fl
from flwr.common import Metrics

from typing import Dict, List, Optional, Tuple, Union
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[
                Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes],
                BaseException,
            ]
        ],
    ) -> Optional[fl.common.NDArrays]:
        weights = super().aggregate_fit(server_round, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {server_round} weights...")
            np.savez(f"round-{server_round}-weights.npz", *weights)
        return weights
    
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Thist function averages teh `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # Number of local epochs done by clients
        "batch_size": 16,  # Batch size to use by clients during fit()
    }
    return config

def main() -> None:
    """Start server and train five rounds."""
    # Load evaluation data
    args = parser.parse_args()

    print(args)

     # Create client_manager, strategy, and server
    strategy = SaveModelStrategy(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Run server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()