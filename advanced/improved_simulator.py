import os
import json
import random
import argparse
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Universal Problem Pool (Math, Coding, Reasoning, Science)
PROBLEMS = [
    {
        "domain": "Math",
        "description": "Carla is downloading a 200 GB file at 2 GB/min. At 40%, Windows forces a 20-min update. She restarts from 0. How long does it take?",
        "mistake": "The student forgets she restarted from 0% and just adds the remaining 60% time onto the restart time."
    },
    {
        "domain": "Coding",
        "description": "Write a Python function to reverse a string without using the [::-1] slicing shortcut.",
        "mistake": "The student tries to use a for loop but iterates out of bounds (IndexError) by going up to len(s) instead of len(s)-1."
    },
    {
        "domain": "Reasoning",
        "description": "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
        "mistake": "The student thinks 'no' because they confuse 'all bloops are lazzies' with 'all lazzies are bloops'."
    },
    {
        "domain": "Science",
        "description": "Explain why ice floats on water in terms of molecular density.",
        "mistake": "The student assumes solid things are always heavier than liquid things and thinks it's floating strictly because of air bubbles trapped inside."
    }
]

SIMULATION_PROMPT = """
Generate a dialogue between an AI tutor and a student.
The student struggles with the following problem and makes a specific mistake.

Problem Description: {description}
Student's Core Mistake: {mistake}

The tutor should perform information-gathering, guiding the student without giving away the final answer immediately. 
In the dialogue, the tutor's utterances are prefaced by "Tutor:" and the student's by "Student:". 

End the dialogue when the student finally reaches the correct conceptual understanding and solves the problem.
The student may get frustrated or easily distracted. The tutor must maintain focus.
"""

def generate_conversation(client: Groq, prob):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are simulating a multi-turn educational dialogue."},
                {"role": "user", "content": SIMULATION_PROMPT.format(description=prob["description"], mistake=prob["mistake"])}
            ],
            temperature=0.8,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_conversation(raw_text):
    turns = []
    current_role = None
    current_text = []
    for line in raw_text.strip().split('\n'):
        if line.startswith('Tutor:'):
            if current_role: turns.append({"role": current_role, "text": ' '.join(current_text).strip()})
            current_role = 'Tutor'
            current_text = [line.replace('Tutor:', '').strip()]
        elif line.startswith('Student:'):
            if current_role: turns.append({"role": current_role, "text": ' '.join(current_text).strip()})
            current_role = 'Student'
            current_text = [line.replace('Student:', '').strip()]
        elif current_role:
            current_text.append(line.strip())
    if current_role: turns.append({"role": current_role, "text": ' '.join(current_text).strip()})
    return turns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=8, help="Number of multi-domain synthetic conversations")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    os.makedirs("advanced", exist_ok=True)
    
    if not api_key:
        print("GROQ_API_KEY not set. Generating stub data for testing logic across domains.")
        # We will create one stub per domain
        stub_data = []
        for p in PROBLEMS:
            stub_data.append({
                "problem": p["description"],
                "dialogue": [
                    {"role": "Student", "text": "I can't figure this out."},
                    {"role": "Tutor", "text": "Let's take it step by step. What do you know?"},
                    {"role": "Student", "text": "Oh okay, I think I have it now! The answer is obvious!"}
                ]
            })
        with open("advanced/cross_domain_dataset.json", "w") as f:
            json.dump(stub_data, f, indent=2)
        print("Saved multi-domain stub dataset to advanced/cross_domain_dataset.json")
        return

    client = Groq(api_key=api_key)
    dataset = []
    
    for _ in tqdm(range(args.num)):
        prob = random.choice(PROBLEMS)
        raw_convo = generate_conversation(client, prob)
        if raw_convo:
            parsed = parse_conversation(raw_convo)
            dataset.append({
                "problem": prob["description"],
                "dialogue": parsed
            })
            
    with open("advanced/cross_domain_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    main()
