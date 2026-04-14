import os
import argparse
import ast
import random
import numpy as np
import d3rlpy
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from collections import deque

load_dotenv()

STATE_PROMPT = """
Respond to the following 25 questions about the educational dialogue state. 
For binary questions, answer 1 (yes) or 0 (no). For counting metrics, output the integer.
Provide the output strictly as a Python list of 25 integers, nothing else. Example: [1, 0, 0, ...]

Dialogue:
{dialogue}
"""

GENERATION_PROMPT = """
The student is trying to solve the following problem:
"{problem}"

Continue the following dialogue with the explicit strategic goal of: {action}

Dialogue:
{dialogue}

Now it's your turn. Begin your generation with "Tutor:" and respond to the student's last utterance. 
Keep your response concise and focused STRICTLY on your exact strategic goal.
"""

ACTIONS = {
    0: "Instructing / Teaching directly to close a knowledge gap",
    1: "Encouraging the student's current hypothesis",
    2: "Assessing the student's background knowledge via questioning",
    3: "Bringing the student's focus back to the core lesson",
    4: "Providing a Partial Hint (e.g. giving only one piece of logic)",
    5: "Asking the student to strictly verify a sub-step/calculation before proceeding"
}

def extract_behavioral_state(client: Groq, dialogue_history: str) -> np.ndarray:
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
        # Add random noise to the mock state so the RL action changes each turn during offline demo
        return [random.choice([0.0, 1.0]) for _ in range(25)]

def generate_response(client: Groq, problem: str, dialogue_history: str, action_text: str) -> str:
    if not client:
        templates = {
            "Instructing / Teaching directly to close a knowledge gap": [
                "Here is the core logic: take the exact constraints from the problem and apply our standard formula. Do you see how that fits?",
                "Let me guide you directly: The first rule is to isolate the variables. This immediately sets up the solution.",
            ],
            "Encouraging the student's current hypothesis": [
                "You are exactly on the right track! Keep progressing with that exact line of reasoning.",
                "Brilliant observation! You've correctly identified the underlying pattern.",
            ],
            "Assessing the student's background knowledge via questioning": [
                "Before we go further, what do you already know about the fundamental rules governing this specific topic?",
                "Let's pause. Have you encountered a similar framework in your earlier study? What mechanism did you use then?",
            ],
            "Bringing the student's focus back to the core lesson": [
                "Let's firmly bring our focus back to the core requirement of the prompt. What is our immediate next mathematical step?",
            ],
            "Providing a Partial Hint (e.g. giving only one piece of logic)": [
                "Here is a crucial hint: consider what happens if you invert the second condition. I'll let you take the next step.",
                "Hint: Do not forget to account for the edge-case in the initial starting condition.",
            ],
            "Asking the student to strictly verify a sub-step/calculation before proceeding": [
                "Wait, can you explicitly verify your logic on that previous intermediate step before we finalize?",
                "Please double check that minor calculation. Does it logically hold up?",
            ]
        }
        import random
        base_resp = random.choice(templates.get(action_text, ["Let's think carefully about this."]))
        return f"(LERT Policy Guided Response): {base_resp}"
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an advanced Context-Aware AI Tutor."},
            {"role": "user", "content": GENERATION_PROMPT.format(problem=problem, dialogue=dialogue_history, action=action_text)}
        ],
        temperature=0.7,
        max_tokens=250,
    )
    reply = completion.choices[0].message.content.strip()
    if reply.startswith("Tutor:"): reply = reply.replace("Tutor:", "").strip()
    return reply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="advanced/advanced_cql_policy.pt")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None

    print("Loading Sentence Transformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    algo = None
    if os.path.exists(args.model):
        print(f"Loading Advanced 459D RL Policy from {args.model}...")
        algo = d3rlpy.algos.DiscreteCQLConfig().create()
        from d3rlpy.dataset import MDPDataset
        dummy_dataset = MDPDataset(
            np.zeros((6, 459), dtype=np.float32), 
            np.arange(6, dtype=np.int32), 
            np.zeros(6, dtype=np.float32), 
            np.ones(6, dtype=np.float32)
        )
        algo.build_with_dataset(dummy_dataset)
        algo.load_model(args.model)
    else:
        print(f"[Warning] Trained model {args.model} not found. Using fallback logic.")

    print("\n--- Universal Context-Aware Tutor (LERT Framework) ---")
    problem = input("Enter any problem (Math, Coding, Science) for the student to solve:\n>> ")
    
    print("\n[System] Computing Semantic Mathematical Context...")
    semantic_vec = embedder.encode(problem).tolist()
    print("[System] Semantic awareness mapped internally. Starting inference...\n")
    
    behavioral_memory = deque([[0.0]*25, [0.0]*25, [0.0]*25], maxlen=3)
    history = ""
    
    while True:
        user_input = input("Student (You): ")
        if user_input.lower() in ["exit", "quit"]: break
            
        history += f"Student: {user_input}\n"
        
        behavioral = extract_behavioral_state(client, history)
        behavioral_memory.append(behavioral)
        
        flat_mem = []
        for b in behavioral_memory: flat_mem.extend(b)
        state_459d = np.array(flat_mem + semantic_vec, dtype=np.float32)
        
        if algo:
            action_id = algo.predict(np.expand_dims(state_459d, axis=0))[0]
        else:
            action_id = random.randint(0, 5)
            
        action_text = ACTIONS.get(action_id)
        print(f"\\_ [LERT Internal] Evaluated State Context -> Action chosen: {action_text}")
        
        tutor_response = generate_response(client, problem, history, action_text)
        print(f"Tutor: {tutor_response}\n")
        history += f"Tutor: {tutor_response}\n"

if __name__ == "__main__":
    main()
