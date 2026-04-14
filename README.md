# LERT: Latent-Embedded RL Tutor

## Overview
This project presents an advanced optimization of the AI tutoring framework originally introduced in *Efficient Reinforcement Learning for Optimizing Multi-turn Student Outcomes*.

The core objective is to solve the **Generalizability Crisis**. The baseline model successfully learned pedagogical action paths via offline Conservative Q-Learning (CQL). However, its simplistic 25-dimensional behavioral state matrix made it completely "Problem-Agnostic." When evaluated on novel math or coding problems, the baseline success rate dropped from 60% to severely constrained Prompt Engineering levels (~27%).

## The Advanced "Context-Aware" Architecture
To overcome this without introducing massive latency bottlenecks, we have developed the Latent-Embedded RL Tutor (LERT) framework, implementing the following core features:

1. **Semantic Problem Embeddings (409D State):** 
   Using the `SentenceTransformer` (`all-MiniLM-L6-v2`) NLP pipeline, we capture deep mathematical and contextual reasoning by calculating a 384D semantic vector of any requested problem. 
   
2. **Temporal Frame Stacking (Curing Amnesia):**
   By concatenating the 384D embedding with the 3 most recent contiguous 25D emotional/behavioral vectors (`384 + (25 × 3) = 459D`), the network develops immediate temporal awareness without computationally expensive Self-Attention over long sequences.

3. **Dense Step-Wise shaping & Granular Actions:**
   Expanded the agent's action topology from 4 generic labels to 6 highly discrete pedagogical tools (including *'Verify Sub-step'* and *'Partial Hint'*), actively guiding logic without spoonfeeding answers.

## File Structure

- `/data` & `/extractor` & `/training` & `/inference`: Contains the exact strict replication of the original research parameter space (Baseline).
- `/advanced`: Contains the refactored **LERT framework**.
  - `improved_simulator.py`: Implements domain-agnostic synthetic data generation. (Includes LLM-as-a-judge dataset filtering paradigms to prevent hallucination noise).
  - `improved_extractor.py`: Fuses local NLP transformers to append semantic complexity into the state observation matrix.
  - `improved_train.py`: Trains the PyTorch D3RLPY architecture on the dense continuous 459D matrix.
  - `improved_repl.py`: An interactive, purely local inference module serving offline pedagogical responses strictly bounded by offline RL policy selections.

## How to Run Demo

To evaluate the Context-Aware Tutor securely without requiring external LLM API configurations:

```shell
# 1. Boot up the inference module
python advanced/improved_repl.py

# 2. Insert any generic question at the prompt (e.g. Science, Coding, Math)
>> Explain why water expands when it freezes.

# 3. Supply a dummy human response and view the dynamic RL Pedagogic Strategy output instantly.
```
