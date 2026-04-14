import os
import argparse
import ast
import random
import numpy as np
import d3rlpy
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

STATE_PROMPT = """
We are evaluating a dialogue between an online tutor and a sixth-grade student.
Respond to the following 25 questions about the dialogue state. 
For binary questions, answer 1 (yes) or 0 (no). For counting metrics, output the integer.
Provide the output strictly as a Python list of 25 integers, nothing else. Example: [1, 0, 0, ...]

Dialogue:
{dialogue}
"""

GENERATION_PROMPT = """
You are an online math tutor working with a sixth-grade student. Continue the following dialogue with the goal of {action}. 
In the dialogue below, the tutor's utterances are prefaced by "Tutor:" and the sixth-grade student's utterances are prefaced by "Student:". 

{dialogue}

Now it's your turn. Begin your generation with "Tutor:" and respond to the student's last utterance. Keep your response concise and focused strictly on the goal action.
"""

ACTIONS = {
    0: "teaching the student",
    1: "encouraging the student",
    # Mapped from background knowledge / questioning logic
    2: "assessing the student's mathematical background knowledge through a question",
    3: "bringing the student's focus back to the lesson"
}

def extract_state(client: Anthropic, dialogue_history: str) -> np.ndarray:
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=150,
            temperature=0.0,
            system="You are an objective grader outputting python lists.",
            messages=[{"role": "user", "content": STATE_PROMPT.format(dialogue=dialogue_history)}]
        )
        content = response.content[0].text.strip()
        state_list = ast.literal_eval(content)
        return np.array([float(x) for x in state_list], dtype=np.float32)
    except Exception as e:
        print(f"[Warning] Failed to extract state via API, padding with zeros: {e}")
        return np.zeros(25, dtype=np.float32)

def generate_response(client: Anthropic, dialogue_history: str, action_text: str) -> str:
    if not client:
        return f"[Simulated Response for Action: {action_text}] Let's try looking at the problem again!"
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=250,
        temperature=0.7,
        system="You are a helpful math tutor generating the next response based on a specific high-level action.",
        messages=[{"role": "user", "content": GENERATION_PROMPT.format(dialogue=dialogue_history, action=action_text)}]
    )
    # the LLM might prefix with "Tutor: " so strip it out if present
    reply = response.content[0].text.strip()
    if reply.startswith("Tutor:"):
        reply = reply.replace("Tutor:", "").strip()
    return reply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="training/cql_policy.pt")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else None

    # Try loading the trained model
    algo = None
    if os.path.exists(args.model):
        print(f"Loading RL Policy from {args.model}...")
        # d3rlpy v2 requires building the algorithm first
        algo = d3rlpy.algos.DiscreteCQLConfig().create()
        algo.build_with_env(d3rlpy.envs.DiscreteEnvConfig(observation_space=d3rlpy.envs.BoxSpace(low=-np.inf, high=np.inf, shape=(25,)), action_space=d3rlpy.envs.DiscreteActionSpace(size=4)))
        algo.load_model(args.model)
    else:
        print(f"[Warning] Trained model {args.model} not found. Using random action baseline for REPL.")

    print("\n--- Starting RL Math Tutor (Paper Replica) ---")
    print("Type 'exit' to end the session.\n")
    
    problem = "A merchant buys electronic gadgets worth $8,000. Market rise 1.2% within a month. How much profit? Answer: 96"
    print(f"Current Target Problem: {problem}\n")

    history = ""
    
    while True:
        user_input = input("Student (You): ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        history += f"Student: {user_input}\n"
        
        # 1. State extraction (The heavy LLM step)
        state_vector = extract_state(client, history)
        
        # 2. RL Action prediction
        if algo:
            action_id = algo.predict(np.expand_dims(state_vector, axis=0))[0]
        else:
            action_id = random.randint(0, 3)
            
        action_text = ACTIONS.get(action_id, "teaching the student")
        print(f"[RL Agent Internal Decision -> High-level Action chosen: {action_text}]")
        
        # 3. Generating conditional tutoring text
        tutor_response = generate_response(client, history, action_text)
        print(f"Tutor: {tutor_response}\n")
        
        history += f"Tutor: {tutor_response}\n"

if __name__ == "__main__":
    main()
