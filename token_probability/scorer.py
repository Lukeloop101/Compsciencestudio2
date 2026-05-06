from openai import OpenAI
import math
import numpy as np
import re

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def clean(text):
    return re.sub(r"[^\w\s]", "", text.strip().lower())


def get_token_prob_score(question: str) -> dict:
    response = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant. Always answer with only the answer itself, one word or short phrase, nothing else.",
            },
            {"role": "user", "content": question},
        ],
        logprobs=True,
        max_tokens=10,
        temperature=0,
        top_p=1,
    )

    raw_answer = response.choices[0].message.content
    generated = clean(raw_answer)

    tokens = response.choices[0].logprobs.content

    # Filter bad tokens
    logprobs = [
        t.logprob for t in tokens if t.logprob is not None and t.token.strip() != ""
    ]

    # Remove first-token bias
    if len(logprobs) > 1:
        logprobs = logprobs[1:]

    # Limit to core tokens (optional but useful)
    logprobs = logprobs[:5]

    # Safety
    if len(logprobs) == 0:
        avg_logprob = -999
        min_logprob = -999
        std_logprob = 0
    else:
        avg_logprob = np.mean(logprobs)
        min_logprob = np.min(logprobs)
        std_logprob = np.std(logprobs)

    avg_prob = math.exp(avg_logprob)
    uncertainty = 1 - avg_prob

    return {
        "question": question,
        "raw_answer": raw_answer,
        "generated_answer": generated,
        "avg_logprob": round(avg_logprob, 4),
        "avg_prob": round(avg_prob, 4),
        "min_logprob": round(min_logprob, 4),
        "std_logprob": round(std_logprob, 4),
        "uncertainty_score": round(uncertainty, 4),
    }
