import os
import re
import pandas as pd
from datasets import load_dataset
from scorer import get_token_prob_score


def clean(text):
    return re.sub(r"[^\w\s]", "", text.strip().lower())


dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation[:200]")

results = []

for i, item in enumerate(dataset):
    q = item["question"]
    ground_truth = item["answer"]["value"]

    print(f"[{i + 1}/200] {q}")

    result = get_token_prob_score(q)

    gt_clean = clean(ground_truth)
    pred_clean = clean(result["generated_answer"])

    result["ground_truth"] = gt_clean

    result["is_correct"] = (
        gt_clean == pred_clean or gt_clean in pred_clean or pred_clean in gt_clean
    )

    if not result["generated_answer"]:
        result["generated_answer"] = "EMPTY"

    results.append(result)

# Save
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)

df["model"] = "llama3.1:8b"
df["dataset_split"] = "validation[:200]"

df.to_csv("results/token_prob_results.csv", index=False)

print("Done! Results saved to results/token_prob_results.csv")

# Quick evaluation
print("\n--- Summary ---")
print(df.groupby("is_correct")["uncertainty_score"].mean())

print("\nCounts:")
print(df["is_correct"].value_counts())
