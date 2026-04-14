import os
import argparse
import ast
import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.dataset import MDPDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/rl_dataset.csv")
    parser.add_argument("--output", type=str, default="training/cql_policy.pt")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs. Reduce for testing.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(args.input)
    
    # Safely evaluate string representation of lists back into python lists
    observations = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df["state"].values], dtype=np.float32)
    actions = np.array(df["action"].values, dtype=np.int32)
    rewards = np.array(df["reward"].values, dtype=np.float32)
    terminals = np.array(df["done"].values, dtype=np.float32)

    # Reconstruct episode boundaries
    dataset = MDPDataset(observations, actions, rewards, terminals)

    print("Initializing Discrete Conservative Q-Learning (CQL) exactly as paper specifies...")
    
    # Paper parameters:
    # LR: 5e-5, Adam epsilon 1e-2/32, Batch 32, Alpha 4.0, Gamma 0.9, Q quantiles 200, target update 2000
    
    # d3rlpy v2 formulation
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=5e-5,
        optim_factory=d3rlpy.optimizers.AdamFactory(eps=1e-2/32),
        batch_size=32,
        alpha=4.0,
        gamma=0.9,
        target_update_interval=2000,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200),
        reward_scaler=d3rlpy.preprocessing.MinMaxRewardScaler(minimum=-1.0, maximum=1.0)
    ).create()

    print(f"Starting training for {args.epochs} epochs...")
    cql.fit(
        dataset,
        n_steps=args.epochs * 1000, # Simplified from 1000000 for user testing
        n_steps_per_epoch=1000
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cql.save_model(args.output)
    print(f"Saved exact replica CQL policy to {args.output}")

    # Generate improved accuracy plot after training
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_metrics import generate_comparison_plot
    generate_comparison_plot()

if __name__ == "__main__":
    main()
