from datasets import load_dataset
from tokenizer import get_tokenizer, tokenize_dataset
from model import load_model
from train import train_model
from evaluate import evaluate_model
from save_model import save_model

dataset = load_dataset('imdb')

tokenizer = get_tokenizer()
tokenized_dataset = tokenize_dataset(dataset, tokenizer)

model = load_model()

trainer = train_model(model, tokenized_dataset)

eval_results = evaluate_model(trainer)
print("Evaluation results:", eval_results)

save_model(model, tokenizer)