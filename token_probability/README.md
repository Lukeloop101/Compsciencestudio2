# Token Probability Method

Measures hallucination uncertainty by averaging log probabilities across output tokens.
Higher uncertainty score = model was less confident = more likely hallucinating.

## Results Summary

- Dataset: TriviaQA (rc.nocontext), 200 validation questions
- Model: llama3.1:8b (local via Ollama)
- Accuracy: 62% (124/200 correct)
- AUROC: 0.667
- Avg uncertainty correct answers: 0.139
- Avg uncertainty incorrect answers: 0.241

## Setup

### 1. Install Ollama (Linux/WSL)

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b

### 2. Python setup

From the root of the repo:

    source venv/bin/activate
    pip install -r requirements.txt

### 3. Run

    cd token_probability
    python run.py
    python analyse.py

Results saved to token_probability/results/token_prob_results.csv

## Output CSV Format

| Column | Type | Description |
|---|---|---|
| question | str | Input question from TriviaQA |
| raw_answer | str | Model raw response before cleaning |
| generated_answer | str | Cleaned model response |
| avg_logprob | float | Average log prob, negative, closer to 0 = confident |
| avg_prob | float | Average probability 0-1, higher = more confident |
| min_logprob | float | Lowest log prob token |
| std_logprob | float | Std deviation of log probs |
| uncertainty_score | float | Final uncertainty 0-1, higher = more uncertain |
| ground_truth | str | Correct answer from dataset |
| is_correct | bool | Whether model answered correctly |
| model | str | Model used |
| dataset_split | str | Dataset split used |

## For pipeline/combine.py

Minimum columns needed to merge with other methods:

- question
- uncertainty_score (normalised 0-1)
- is_correct