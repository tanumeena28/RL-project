import os
import argparse
import ast
import numpy as np
import pandas as pd
import d3rlpy
from d3rlpy.dataset import MDPDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="advanced/advanced_dataset.csv")
    parser.add_argument("--output", type=str, default="advanced/advanced_cql_policy.pt")
    # Using more epochs by default to properly train the denser shaping
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print("Loading 459D Semantic/Temporal dataset...")
    df = pd.read_csv(args.input)
    
    observations = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in df["state"].values], dtype=np.float32)
    actions = np.array(df["action"].values, dtype=np.int32)
    rewards = np.array(df["reward"].values, dtype=np.float32)
    terminals = np.array(df["done"].values, dtype=np.float32)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    print("Initializing Multi-Domain CQL Model with Granular Action Scaler...")
    
    # We maintain Alpha=4.0 and Gamma=0.9, but adjust to expect the larger continuous bounds
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=5e-5,
        optim_factory=d3rlpy.optimizers.AdamFactory(eps=1e-2/32),
        batch_size=32,
        alpha=4.0,
        gamma=0.9,
        # Action discrete size automatically inferred from data as 6 by builder
    ).create()

    print(f"Starting generic cross-domain training for {args.epochs} epochs over 459 dimensions...")
    cql.fit(
        dataset,
        n_steps=args.epochs * 1000, 
        n_steps_per_epoch=1000
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cql.save_model(args.output)
    print(f"Saved exact replica CQL policy to {args.output}")

if __name__ == "__main__":
    main()
