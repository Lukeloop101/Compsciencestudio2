import json
import re
import pandas as pd
from groq import Groq
from datasets import load_dataset
import matplotlib.pyplot as plt
from openai import OpenAI

#client = Groq(api_key="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")
#May need to deal with local host
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def clean(text):
    if text is None:
        return ""
    return re.sub(r"[^\w\s]", "", str(text).strip().lower())


def is_match(pred, gt):
    if not pred or not gt:
        return False
    return pred == gt or pred in gt or gt in pred


def parse_json(text):
    try:
        return json.loads(text)
    except:
        return None


def ask_model(prompt, temperature=0):
    response = client.chat.completions.create(
        #if change local may need to do something
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "Return only strict JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def build_answer_prompt(question):
    return f"""
Answer the question given as well as a confidence score from 0.0 to 1.0.

rules for different confidence levels:
- 0.95 to 1.00 = only if you are certain the answer is correct
- 0.80 to 0.94 = strong evidence, but not absolute certainty
- 0.50 to 0.79 = plausible guess, but uncertain
- 0.20 to 0.49 = weak guess
- 0.00 to 0.19 = almost no confidence

Important rules:
- Do NOT give confidence above 0.90 unless the answer is extremely well-known and you are certain.
- If there is any ambiguity, lower the confidence.
- If you are guessing, confidence must be below 0.70.
- If you are not sure, prefer a lower confidence rather than a higher one.
- Most of the answers confidence score should not be above 0.90.

Return ONLY valid JSON:
{{
  "answer": "short answer",
  "confidence": 0.67
}}

Question: {question}
"""


dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation[:200]")

results = []
correct_count = 0
total = len(dataset)

for i, item in enumerate(dataset):
    question = item["question"]
    ground_truth = clean(item["answer"]["value"])

    print("\n====================")
    print(f"Question {i+1}: {question}")
    print("Ground truth:", ground_truth)

    answer_raw = ask_model(build_answer_prompt(question), temperature=0.5)
    answer_data = parse_json(answer_raw)

    if not answer_data:
        print("Failed to parse answer JSON")
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": "",
            "confidence": 0,
            "is_correct": False,
            "parse_failed": True
        })
        continue

    model_answer = clean(answer_data.get("answer", ""))
    confidence = answer_data.get("confidence", 0)

    try:
        confidence = float(confidence)
    except:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    correct = is_match(model_answer, ground_truth)

    if correct:
        correct_count += 1

    print("Model answer:", model_answer)
    print("Confidence:", confidence)
    print("Correct" if correct else "Incorrect")

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "confidence": confidence,
        "is_correct": correct,
        "parse_failed": False
    })

df = pd.DataFrame(results)
df.to_csv("trivia_self_confidence_results.csv", index=False)

print("\n====================")
print(f"Final Score: {correct_count}/{total}")
print(f"Accuracy: {correct_count / total * 100:.2f}%")
print("Saved to trivia_self_confidence_results.csv")




#HE
# ----------------------------
# Plot CONFIDENCE distribution
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

df[df["is_correct"]]["confidence"].hist(
    bins=20, alpha=0.6, label="Correct", ax=ax
)

df[~df["is_correct"]]["confidence"].hist(
    bins=20, alpha=0.6, label="Incorrect", ax=ax
)

ax.set_xlabel("Confidence Score")
ax.set_ylabel("Count")
ax.set_title("Self-Verbalised Confidence Distribution")
ax.legend()

plt.tight_layout()
plt.savefig("trivia_confidence_distribution.png")
plt.show()

print("Plot saved to trivia_confidence_distribution.png")
