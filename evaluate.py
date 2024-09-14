def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    return eval_results