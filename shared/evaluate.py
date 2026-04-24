from responseconsistency import analyse_question
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scorer import get_token_prob_score

#my goal is first to figure out how the differnt models are answering the questions, and then to figure out how to evaluate the answers. I will start with the first part, and then move on to the second part.

# Keep this small at first so runtime is manageable the dataset 
dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation[:200]")    


questions = []

for item in dataset:
    questions.append({
        "question": item["question"],
        "answer": item["answer"]["value"],
        "aliases": item["answer"]["aliases"]
    })
    
    #realistically add in alisies later
resultsConsistency = []
resultsSelf = []
resultsToken = []
for i, item in enumerate(questions, start=1):
    question = item["question"]
    answer = item["answer"]
    aliases = item["aliases"]
    
    print(f"[{i + 1}/200] {question}")
    
    # This is the entire conistency for each with long output recheck
    resultsConsistency = analyse_question(question, answer, aliases, n=10)
    
    
    # self
    answer_raw = ask_model(build_answer_prompt(question), temperature=0.5)
    answer_data = parse_json(answer_raw)
    
    model_answer = clean(answer_data.get("answer", ""))
    confidence = answer_data.get("confidence", 0)
    
    #token
    #hmm its result vs results
    result = get_token_prob_score(question)
    
    #"question": question," raw_answer": raw_answer, "generated_answer": generated,"avg_logprob": round(avg_logprob, 4),"avg_prob": round(avg_prob, 4), "min_logprob": round(min_logprob, 4), "std_logprob": round(std_logprob, 4), "uncertainty_score": round(uncertainty, 4),
    
    #IT was done backwards so need to un backwards it
    
    
    #Now we should be able to compare the results and conbine them
    