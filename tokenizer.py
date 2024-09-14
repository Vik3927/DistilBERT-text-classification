from transformers import AutoTokenizer

def get_tokenizer(model_name='distilbert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset