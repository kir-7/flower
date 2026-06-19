#!/bin/bash

echo "Experiment 1: FedAvg | fraction-train=0.5"
flwr run . --run-config "algorithm='fedavg' fraction-train=0.5 lr=0.01 proximal_mu=0.0 save-dir='fedavg_results/'" --stream
echo

echo "Experiment 3: FedProx | fraction-train=0.5 | proximal_mu=0.6"
flwr run . --run-config "algorithm='fedavg' fraction-train=0.5 lr=0.01 proximal_mu=0.6 save-dir='fedprox_results/'" --stream
echo


echo "Experiment 5: FedAMP | fraction-train=0.5"
flwr run . --run-config "algorithm='fedamp' fraction-train=0.5 lr=0.01 alphaK=1.0 fedamp-lambda=1.0 sigma=1.0 save-dir='fedamp_results/'" --stream
echo

