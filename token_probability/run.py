import os
import re
import pandas as pd
from datasets import load_dataset
from scorer import get_token_prob_score


def clean(text):
    return re.sub(r"[^\w\s]", "", text.strip().lower())


def is_match(pred, alias):
    # same exact-or-substring check as before, just pulled out so we can run it
    # against every alias without making the loop messy
    if not alias or not pred:
        return False
    return pred == alias or alias in pred or pred in alias


N = 3000
SPLIT = f"validation[:{N}]"

dataset = load_dataset("trivia_qa", "rc.nocontext", split=SPLIT)
results = []

for i, item in enumerate(dataset):
    q = item["question"]
    ans = item["answer"]

    # main ground truth, kept for the csv so the output looks like before
    ground_truth = ans["value"]

    # gather acceptable answers from all four fields trivia_qa provides.
    # "value" and "aliases" are raw, "normalized_*" are pre-cleaned
    aliases = set()
    aliases.add(ans.get("value", "") or "")
    aliases.update(ans.get("aliases", []) or [])
    aliases.add(ans.get("normalized_value", "") or "")
    aliases.update(ans.get("normalized_aliases", []) or [])

    # clean each one the same way as the prediction so comparison is fair
    aliases_clean = {clean(a) for a in aliases if a}

    print(f"[{i + 1}/{N}] {q}")
    result = get_token_prob_score(q)

    pred_clean = clean(result["generated_answer"])

    # correct if it matches ANY alias - fixes the WWII vs World War Two
    # problem we hit on the 200 run
    result["ground_truth"] = clean(ground_truth)
    result["num_aliases"] = len(aliases_clean)
    result["is_correct"] = any(is_match(pred_clean, a) for a in aliases_clean)

    if not result["generated_answer"]:
        result["generated_answer"] = "EMPTY"
    results.append(result)

# save
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df["model"] = "llama3.1:8b"
df["dataset_split"] = SPLIT
df.to_csv("results/token_prob_results.csv", index=False)
print(f"Done! Results saved to results/token_prob_results.csv ({len(df)} rows)")

# quick sanity check
print("\n--- Summary ---")
print(df.groupby("is_correct")["uncertainty_score"].mean())
print("\nCounts:")
print(df["is_correct"].value_counts())
print(f"\nAccuracy: {df['is_correct'].mean():.3f}")