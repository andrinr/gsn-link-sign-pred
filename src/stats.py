# External dependencies
import nevergrad as ng
import sys, getopt
import torch
import numpy as np
import torch_geometric.transforms as T
import csv
# Local dependencies
from model import Training
from data import WikiSigned, Tribes, Chess, BitcoinA

def main(argv) -> None:
    
    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    parametrization = ng.p.Instrumentation(
        friend_distance=ng.p.Scalar(lower=0.1, upper=20.0),
        friend_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
        neutral_distance=ng.p.Scalar(lower=0.1, upper=20.0),
        neutral_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
        enemy_distance=ng.p.Scalar(lower=0.1, upper=20),
        enemy_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100, num_workers=1)

    recommendation = optimizer.minimize(training)

    return recommendation

if __name__ == "__main__":
    main(sys.argv[1:])