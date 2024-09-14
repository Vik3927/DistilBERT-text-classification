from transformers import Trainer, TrainingArguments

def train_model(model, tokenized_dataset, output_dir='./results', batch_size=8, epochs=3, weight_decay=0.01):
    training_args = TrainingArguments(
        output_dir=output_dir,          
        evaluation_strategy="epoch",    
        per_device_train_batch_size=batch_size,   
        per_device_eval_batch_size=batch_size,    
        num_train_epochs=epochs,              
        weight_decay=weight_decay,               
    )
    
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_dataset['train'],         
        eval_dataset=tokenized_dataset['test'],            
    )
    
    trainer.train()
    return trainer