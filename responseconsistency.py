from openai import OpenAI
from config import MODEL_NAME, BASE_URL, API_KEY
import re
from collections import Counter

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)


def ask_model(question: str) -> str:
    """
    Ask the model one question and return a short answer.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Answer briefly in one short phrase only. Do not explain."
            },
            {"role": "user", "content": question}
        ],
        temperature=1.0,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for easier comparison.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def generate_multiple_responses(question: str, n: int = 3) -> list[str]:
    """
    Ask the same question n times and return all responses.
    """
    responses = []
    for _ in range(n):
        responses.append(ask_model(question))
    return responses


def is_answer_correct(answer: str, correct_answer: str, aliases: list[str]) -> bool:
    """
    Check whether a normalized answer matches the correct answer or any alias.
    """
    answer_norm = normalize_text(answer)

    valid_answers = [correct_answer] + aliases
    valid_answers = [normalize_text(a) for a in valid_answers if a]

    for valid in valid_answers:
        if answer_norm == valid:
            return True
        if valid in answer_norm:
            return True

    return False


def analyse_question(question: str, correct_answer: str, aliases: list[str], n: int = 3) -> dict:
    """
    For one question:
    - generate n responses
    - find the most common response
    - compute consistency score = frequency of most common response / n
    - determine whether the most common response is correct
    """
    responses = generate_multiple_responses(question, n)
    normalized_responses = [normalize_text(r) for r in responses]

    counter = Counter(normalized_responses)
    most_common_answer, most_common_count = counter.most_common(1)[0]

    consistency_score = most_common_count / n
    correct = is_answer_correct(most_common_answer, correct_answer, aliases)

    return {
        "question": question,
        "correct_answer": correct_answer,
        "responses": responses,
        "normalized_responses": normalized_responses,
        "most_common_answer": most_common_answer,
        "most_common_count": most_common_count,
        "consistency_score": round(consistency_score, 4),
        "is_correct": correct
    }