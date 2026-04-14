import os
import json
import argparse
from anthropic import Anthropic
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Example math problem to simulate tutoring on (from GSM8K)
SAMPLE_PROBLEM = "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file? Answer: 160."
SAMPLE_MISTAKE = "The student forgets about the 40% of the file downloaded before the restart and only calculates the restart time and the final full download time."

SIMULATION_PROMPT = f"""
Generate a dialogue between an AI tutor and a sixth-grade student where the sixth-grade student asks the tutor for an explanation of a math problem. 
The student is not good at math, so the student struggles with the problem and makes a mistake. 

Here is the student's mistake: {SAMPLE_MISTAKE}

The tutor should perform information-gathering to figure out the sixth-grade student's math background knowledge, by asking questions and engaging in dialogue with the sixth-grade student. 
In the dialogue, the tutor's utterances are prefaced by "Tutor:" and the sixth-grade student's utterances are prefaced by "Student:". 

The student is asking about the following problem:
{SAMPLE_PROBLEM}

Make sure the dialogue ends when the student gives the correct answer. 
The tutor should not give the solution explicitly but correct the student's mistake if the student makes any mistakes. 
The student is easily distracted and may lose interest in solving the problem, but the tutor needs to help the student focus on the problem.
"""

def generate_conversation(client: Anthropic):
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            temperature=0.7,
            system="You are acting as both an AI tutor and a distracted sixth-grade student to generate a simulated dialogue.",
            messages=[
                {"role": "user", "content": SIMULATION_PROMPT}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return None

def parse_conversation(raw_text):
    """Simple parser to convert raw generated text into a list of dictionaries."""
    turns = []
    current_role = None
    current_text = []
    
    for line in raw_text.strip().split('\n'):
        if line.startswith('Tutor:'):
            if current_role:
                turns.append({"role": current_role, "text": ' '.join(current_text).strip()})
            current_role = 'Tutor'
            current_text = [line.replace('Tutor:', '').strip()]
        elif line.startswith('Student:'):
            if current_role:
                turns.append({"role": current_role, "text": ' '.join(current_text).strip()})
            current_role = 'Student'
            current_text = [line.replace('Student:', '').strip()]
        elif current_role:
            current_text.append(line.strip())
            
    if current_role:
        turns.append({"role": current_role, "text": ' '.join(current_text).strip()})
        
    return turns

def main():
    parser = argparse.ArgumentParser(description="Simulate student-tutor conversations using Claude 3 Sonnet.")
    parser.add_argument("--num_convos", type=int, default=5, help="Number of dummy conversations to generate.")
    parser.add_argument("--output", type=str, default="data/dialogues.json", help="Output JSON file.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("For testing without a key, creating a fake stub dataset.")
        # Create a tiny fake dataset to proceed with pipeline construction
        stub_data = [
            [
                {"role": "Student", "text": "I don't know how to do this Carla downloading problem."},
                {"role": "Tutor", "text": "Let's break it down. How many GB is the whole file?"},
                {"role": "Student", "text": "200 GB. And she downloads 2 GB per minute."},
                {"role": "Tutor", "text": "Good! And she got 40% done before restarting. Can you find 40% of 200?"},
                {"role": "Student", "text": "80 GB?"},
                {"role": "Tutor", "text": "Right. How long did that 80 GB take?"},
                {"role": "Student", "text": "80 / 2 = 40 minutes."},
                {"role": "Tutor", "text": "Exactly. Then what happened?"},
                {"role": "Student", "text": "An update took 20 minutes. And she starts over, so she has to download 200 GB again. 200 / 2 = 100 minutes. So 40 + 20 + 100 = 160 minutes!"},
                {"role": "Tutor", "text": "You got it! 160 is the correct answer."}
            ]
        ]
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(stub_data, f, indent=2)
        print(f"Saved stub dataset to {args.output}")
        return

    client = Anthropic(api_key=api_key)
    dataset = []
    
    print(f"Generating {args.num_convos} simulated conversations...")
    for _ in tqdm(range(args.num_convos)):
        raw_convo = generate_conversation(client)
        if raw_convo:
            parsed = parse_conversation(raw_convo)
            dataset.append(parsed)
            
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Successfully saved {len(dataset)} conversations to {args.output}")

if __name__ == "__main__":
    main()
