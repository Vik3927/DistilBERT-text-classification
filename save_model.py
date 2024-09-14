def save_model(model, tokenizer, save_dir='distilbert-imdb'):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)