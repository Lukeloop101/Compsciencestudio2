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

def checkCorrect(ourAnswer, BaseAnswer, question, aliases):
    #used this so it first checks if the same stole from token also saves time and technically tokens but who cares about that.
    if (not(ourAnswer.lower().strip() == BaseAnswer.lower().strip() or ourAnswer.lower().strip() in BaseAnswer.lower().strip() or BaseAnswer.lower().strip() in ourAnswer.lower().strip() or any(ourAnswer.lower().strip() in alias.strip().lower() for alias in aliases ))):
        #Bro PROMPTING IS SO SPECIFIC ITS GHRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
        prompt = prompt = f"""You are checking if a model's answer is correct given a question and known correct answer.
    Question: {question}
    Correct Answer: {BaseAnswer}
    Model Answer: {ourAnswer}

    Return just one word: true or false."""


        response = client.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "You compare answers. Return only true or false. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        output = response.choices[0].message.content

        if "true" in output.lower():
            return True
        elif "false" in output.lower():
            return False
        else:
            return False
    return ourAnswer.lower().strip() == BaseAnswer.lower().strip() or ourAnswer.lower().strip() in BaseAnswer.lower().strip() or BaseAnswer.lower().strip() in ourAnswer.lower().strip() or any(ourAnswer.lower().strip() in alias.strip().lower() for alias in aliases )
# def is_match(pred, gt):
#     if not pred or not gt:
#         return False
#     return pred == gt or pred in gt or gt in pred


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
    return f"""Given the question answer it and provide a confidence score between 0.00 and 1.00
Specific requirements for each level of confidence:
- 0.95 to 1.00 =  You are completely sure that the answer is correct, based on multiple sources.
- 0.80 to 0.94 = strong evidence, but  there is not complete certainty
- 0.50 to 0.79 = There is decent evidence for the answer being correct compared to other choices
- 0.20 to 0.49 =  Weak evidence or close to becoming a guess but the answer  has some merit above others.
- 0.00 to 0.19 = It is practically a guess with very little evidence

Some extra specific rules to follow:
- Do NOT give confidence above 0.90 unless the answer is extremely well-known and you are certain.
- If there is any uncertainity, lower the confidence.
- If you are guessing, confidence must be below 0.70.
- When uncertain of a answer choose a lower score over a higher answer.
- Most of the answers confidence score should not be above 0.90.

Return ONLY valid JSON:
{{
  "answer": "short answer",
  "confidence": 0.67
}}

Question: {question}
"""
#in a main because otherwise it runs the  whole thing when I import didnt know tha pretty cool
if __name__ == "__main__":
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation[:3000]")

    results = []
    correct_count = 0
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        aliases = item["answer"]["aliases"]
        ground_truth = str(item["answer"]["value"]).strip()

        print("\n====================")
        print(f"Question {i+1}: {question}")
        print("Ground truth:", ground_truth)

        
        answer_data = parse_json(ask_model(build_answer_prompt(question), temperature=0.5))

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

        model_answer = str(answer_data.get("answer", "")).strip()
        confidence = answer_data.get("confidence", 0)

        try:
            confidence = float(confidence)
        except:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        correct = checkCorrect(model_answer, ground_truth, question, aliases)

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
