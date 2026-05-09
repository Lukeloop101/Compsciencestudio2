import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("results/token_prob_results.csv")

# the runner already computed alias-aware correctness in `is_correct`,
# so we just use that directly instead of recomputing here

# ----------------------------
# Basic stats
# ----------------------------
total = len(df)
correct = df["is_correct"].sum()
incorrect = total - correct
print(f"Total questions: {total}")
print(f"Correct: {correct} ({correct / total * 100:.1f}%)")
print(f"Incorrect: {incorrect} ({incorrect / total * 100:.1f}%)")

# ----------------------------
# Grouped analysis
# ----------------------------
print("\n--- Uncertainty by correctness ---")
print(
    df.groupby("is_correct")["uncertainty_score"]
    .mean()
    .rename({True: "Correct", False: "Incorrect"})
)

print("\n--- Avg logprob by correctness ---")
print(
    df.groupby("is_correct")["avg_logprob"]
    .mean()
    .rename({True: "Correct", False: "Incorrect"})
)

print("\n--- Min logprob by correctness ---")
print(
    df.groupby("is_correct")["min_logprob"]
    .mean()
    .rename({True: "Correct", False: "Incorrect"})
)

# ----------------------------
# AUROC calculations
# ----------------------------
y_true = df["is_correct"].astype(int)

def safe_auc(score, invert=False):
    s = -score if invert else score
    return roc_auc_score(y_true, s)

auroc_avg_logprob = safe_auc(df["avg_logprob"], invert=False)
auroc_min_logprob = safe_auc(df["min_logprob"], invert=False)
auroc_uncertainty = safe_auc(df["uncertainty_score"], invert=True)

print("\n--- AUROC Scores ---")
print(f"Avg Logprob AUROC: {auroc_avg_logprob:.4f}")
print(f"Min Logprob AUROC: {auroc_min_logprob:.4f}")
print(f"Uncertainty Score AUROC: {auroc_uncertainty:.4f}")

# ----------------------------
# Plot distribution
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))
df[df["is_correct"]]["uncertainty_score"].hist(
    bins=20, alpha=0.6, label="Correct", ax=ax
)
df[~df["is_correct"]]["uncertainty_score"].hist(
    bins=20, alpha=0.6, label="Incorrect", ax=ax
)
ax.set_xlabel("Uncertainty Score")
ax.set_ylabel("Count")
ax.set_title("Token Probability Uncertainty Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("results/uncertainty_distribution.png")
print("\nPlot saved to results/uncertainty_distribution.png")