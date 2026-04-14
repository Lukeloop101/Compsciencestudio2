from responseconsistency import analyse_question
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# Keep this small at first so runtime is manageable
dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation[:20]")

questions = []

for item in dataset:
    questions.append({
        "question": item["question"],
        "answer": item["answer"]["value"],
        "aliases": item["answer"]["aliases"]
    })

correct_consistency_scores = []
incorrect_consistency_scores = []

for i, item in enumerate(questions, start=1):
    question = item["question"]
    answer = item["answer"]
    aliases = item["aliases"]

    print(f"\nRunning question {i}/{len(questions)}: {question}")

    result = analyse_question(question, answer, aliases, n=20)

    print("Correct answer:", result["correct_answer"])
    print("Responses:", result["responses"])
    print("Most common answer:", result["most_common_answer"])
    print("Most common count:", result["most_common_count"])
    print("Consistency score:", result["consistency_score"])
    print("Correct:", result["is_correct"])

    if result["is_correct"]:
        correct_consistency_scores.append(result["consistency_score"])
    else:
        incorrect_consistency_scores.append(result["consistency_score"])

# Histogram like your example
bins = np.linspace(0, 1, 21)

plt.figure(figsize=(9, 5))
plt.hist(correct_consistency_scores, bins=bins, alpha=0.6, label="Correct")
plt.hist(incorrect_consistency_scores, bins=bins, alpha=0.6, label="Incorrect")

plt.xlabel("Consistency Score")
plt.ylabel("Count")
plt.title("Response Consistency Distribution")
plt.legend()
plt.tight_layout()
plt.show()