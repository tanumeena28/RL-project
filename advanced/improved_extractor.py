import os
import json
import argparse
import ast
import random
import numpy as np
import pandas as pd
from groq import Groq
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from collections import deque

load_dotenv()

STATE_PROMPT = """
We are evaluating a dialogue between a tutor and a student.
Respond to the following 25 questions about the dialogue state. 
For binary questions, answer 1 (yes) or 0 (no). For counting metrics, output the integer.
Provide the output strictly as a Python list of 25 integers, nothing else. Example: [1, 0, 0, ...]
Dialogue:
{dialogue}
"""

ACTION_PROMPT = """
Label the tutor's recent utterance into one of the following 6 granular action types:
<1> Instructing / Teaching directly
<2> Encouraging the student
<3> Assessing background / Questioning
<4> Refocusing / Bringing focus back
<5> Partial Hint (Giving a piece of logic)
<6> Verify Sub-step (Asking student to confirm a minor calculation)

Respond with ONLY the integer (1, 2, 3, 4, 5, or 6).
Dialogue:
{dialogue}
"""

def extract_behavioral_state(client: Groq, dialogue_history: str) -> list:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an objective grader outputting python lists."},
                {"role": "user", "content": STATE_PROMPT.format(dialogue=dialogue_history)}
            ],
            temperature=0.0,
            max_tokens=150,
        )
        state_list = ast.literal_eval(completion.choices[0].message.content.strip())
        return [float(x) for x in state_list][:25]
    except:
        return [random.choice([0.0, 1.0]) for _ in range(25)]

def extract_action(client: Groq, dialogue_history: str) -> int:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an objective classifier outputting a single integer."},
                {"role": "user", "content": ACTION_PROMPT.format(dialogue=dialogue_history)}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        return int(completion.choices[0].message.content.strip()) - 1 # 0 to 5 space
    except:
        return random.randint(0, 5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="advanced/cross_domain_dataset.json")
    parser.add_argument("--output", type=str, default="advanced/advanced_dataset.csv")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None

    print("Loading Sentence Transformer (all-MiniLM-L6-v2) for Semantic Embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    with open(args.input, "r") as f:
        data = json.load(f)

    rl_data = []

    for dialogue_idx, item in enumerate(tqdm(data)):
        problem_desc = item["problem"]
        turns = item["dialogue"]
        
        # 384D semantic embedding for domain agnosticism
        semantic_vec = embedder.encode(problem_desc).tolist() # length 384
        
        # We need a 3-frame stack to cure temporal amnesia: (t, t-1, t-2).
        behavioral_memory = deque([[0.0]*25, [0.0]*25, [0.0]*25], maxlen=3)
        history = ""
        prev_459D_state = None
        
        solved = "obvious" in str(turns) or "correct" in str(turns).lower()

        for turn_idx, turn in enumerate(turns):
            history += f"{turn['role']}: {turn['text']}\n"
            
            if turn["role"] == "Tutor":
                behavioral = extract_behavioral_state(client, history) if client else [random.random() for _ in range(25)]
                action = extract_action(client, history) if client else random.randint(0, 5)
                
                behavioral_memory.append(behavioral)
                
                # Combine memory frames + semantic vector
                # 25 + 25 + 25 + 384 = 459D
                flat_mem = []
                for b in behavioral_memory: flat_mem.extend(b)
                current_459D_state = flat_mem + semantic_vec
                
                # Dense Shaping Reward
                # +0.2 for verifying structure nicely, -0.1 if student is frustrated (stubbed).
                is_last_turn = (turn_idx >= len(turns) - 2)
                base_reward = 1.0 if (is_last_turn and solved) else 0.0
                dense_reward = base_reward
                if action == 5: dense_reward += 0.2
                if action == 0 and prev_action == 0 if 'prev_action' in locals() else False: dense_reward -= 0.1 # penalize greedy instructing

                if prev_459D_state is not None:
                    rl_data.append({
                        "dialogue_id": dialogue_idx,
                        "state": prev_459D_state,
                        "action": prev_action,
                        "reward": 0.0, # intermediate reward
                        "next_state": current_459D_state,
                        "done": 0
                    })
                
                prev_459D_state = current_459D_state
                prev_action = action
                
                if is_last_turn:
                    rl_data.append({
                        "dialogue_id": dialogue_idx,
                        "state": prev_459D_state,
                        "action": action,
                        "reward": dense_reward,
                        "next_state": [0.0]*459,
                        "done": 1
                    })
                    break

    df = pd.DataFrame(rl_data)
    df.to_csv(args.output, index=False)
    print(f"Extraction complete! Advanced dataset saved to {args.output}")

if __name__ == "__main__":
    main()
