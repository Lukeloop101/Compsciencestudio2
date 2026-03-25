import pandas as pd

splits = {'train': 'socratic/train-00000-of-00001.parquet', 'test': 'socratic/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])

print(df.head())
print(df.columns)

#this was base set up for loading the data
#from gsm8k as the dataset currently
#may need to install pandas and pyarrow packages from huggingface_hub to run this code

#we need to choose AI model tp use