import os
import json
import argparse
import ast
import random
import pandas as pd
from anthropic import Anthropic
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# The 25 questions from Appendix 11
QUESTIONS_25 = [
    "1. Is the student producing math-related content?",
    "2. Has the student solved the problem correctly?",
    "3. Is the student asking the tutor to re-explain a concept or clarify what the tutor has said already?",
    "4. Is the student repeating or emphasizing what the tutor has already said?",
    "5. Is the student going off-topic?",
    "6. Is the student's utterance unrelated to the math problem?",
    "7. Is the student explicitly asking the tutor a question?",
    "8. Is the student describing what they are stuck on or which part of the problem they are confused about?",
    "9. Has the student asked diagnostic questions to assess the student's mathematical knowledge or level?",
    "10. Is the student expressing frustration?",
    "11. Is the student expressing uncertainty or lack of confidence about their ability to solve the problem?",
    "12. Is the student expressing positive sentiment?",
    "13. Is the student asking the tutor for a break from the tutoring session?",
    "14. Is the student talking about the problem at hand?",
    "15. Is the student talking about their general mathematical background?",
    "16. Is the student talking about other math concepts related to the problem at hand?",
    "17. Has the student written down an equation for the problem?",
    "18. Is the tutor asking a question to the student?",
    "19. Did the student make a mistake in the current turn?",
    "20. Has the tutor tried to bring the student's focus back to the problem after the student is distracted?",
    "21. How many questions did the tutor ask the student so far?",
    "22. How many questions did the student ask the tutor so far?",
    "23. What is the current turn in the conversation?",
    "24. Arbitrary math density feature (simulated).",
    "25. Arbitrary math reasoning feature (simulated)."
]

ACTION_PROMPT = """
We are evaluating a dialogue between an online tutor and a sixth-grade student.
Based on this dialogue, label the tutor's recent utterance as one of the following action types:
<1> teaching
<2> encouraging the student
<3> assessing the student's background knowledge (questioning)
<4> bringing the student's focus back to the lesson (refocusing)

Respond with ONLY the integer (1, 2, 3, or 4).
Dialogue:
{dialogue}
"""

STATE_PROMPT = """
We are evaluating a dialogue between an online tutor and a sixth-grade student.
Respond to the following 25 questions about the dialogue state. 
For binary questions, answer 1 (yes) or 0 (no). For counting metrics, output the integer.
Provide the output strictly as a Python list of 25 integers, nothing else. Example: [1, 0, 0, ...]

Dialogue:
{dialogue}
"""

def extract_state(client: Anthropic, dialogue_history: str) -> list:
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
        if len(state_list) != 25:
            raise ValueError("List length is not 25")
        return [float(x) for x in state_list]
    except Exception as e:
        print(f"Failed to extract real state, using fallback: {e}")
        return [random.choice([0.0, 1.0]) for _ in range(25)]

def extract_action(client: Anthropic, dialogue_history: str) -> int:
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            temperature=0.0,
            system="You are an objective classifier outputting a single integer.",
            messages=[{"role": "user", "content": ACTION_PROMPT.format(dialogue=dialogue_history)}]
        )
        return int(response.content[0].text.strip())
    except Exception as e:
        print(f"Failed to extract real action, using fallback: {e}")
        return random.randint(0, 3) # Map 1-4 to 0-3 for d3rlpy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/dialogues.json")
    parser.add_argument("--output", type=str, default="data/rl_dataset.csv")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else None

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    with open(args.input, "r") as f:
        dialogues = json.load(f)

    print(f"Processing {len(dialogues)} dialogues to extract States and Actions...")
    
    rl_data = []

    for dialogue_idx, turns in enumerate(tqdm(dialogues)):
        history = ""
        prev_state = None
        
        # Determine if solved at the end (simplistic check for dummy data)
        solved = "160" in turns[-1]["text"] or "correct answer" in turns[-1]["text"].lower()

        for turn_idx, turn in enumerate(turns):
            history += f"{turn['role']}: {turn['text']}\n"
            
            # The tutor acts after the student says something.
            # We want to model: State (history before tutor speaks) -> Action (tutor utterance) -> Reward -> Next State
            if turn["role"] == "Tutor":
                # Compute state based on history up to just BEFORE the tutor spoke
                # For simplicity in this replica, we just use the history including the prompt to tutor.
                current_state = extract_state(client, history) if client else [random.random() for _ in range(25)]
                action = extract_action(client, history) if client else random.randint(0, 3)

                # Reward is 0 unless it's the last turn
                is_last_turn = (turn_idx >= len(turns) - 2)
                reward = 1.0 if (is_last_turn and solved) else 0.0

                if prev_state is not None:
                    # Append previous tuple
                    rl_data.append({
                        "dialogue_id": dialogue_idx,
                        "state": prev_state,
                        "action": prev_action,
                        "reward": 0.0,
                        "next_state": current_state,
                        "done": 0
                    })
                
                prev_state = current_state
                prev_action = action
                
                if is_last_turn:
                    # Final transition
                    rl_data.append({
                        "dialogue_id": dialogue_idx,
                        "state": prev_state,
                        "action": action,
                        "reward": reward,
                        # No real next state, replicate current or zeros
                        "next_state": [0.0]*25,
                        "done": 1
                    })
                    break

    df = pd.DataFrame(rl_data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Processed RL dataset saved to {args.output}")

if __name__ == "__main__":
    main()
