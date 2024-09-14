# DistilBERT Text Classification

This project implements binary text classification using the pre-trained DistilBERT model. The task is to classify text reviews from the IMDb dataset as either **positive** or **negative**. The project is structured to be used in Google Colab, but it can also be adapted to run locally.

## Project Structure

``` DistilBERT-text-classification/
│
├── main.py               # Main script that orchestrates the whole process
├── tokenizer.py          # Script to handle tokenization
├── model.py              # Script to load the pre-trained model
├── train.py              # Script for training the model
├── evaluate.py           # Script for evaluating the model
├── save_model.py         # Script to save the trained model
├── requirements.txt      # Dependencies for the project
└── README.md             # Project documentation
```


## Dataset

The IMDb movie reviews dataset is used for binary classification (positive or negative sentiment). It is automatically downloaded from the Hugging Face `datasets` library.

- **Training Data**: Contains 25,000 labeled movie reviews for training.
- **Test Data**: Contains 25,000 labeled movie reviews for evaluation.

## Model

The project uses the **DistilBERT** model, which is a smaller and faster version of the BERT model. The model is pre-trained on a large corpus of English text and fine-tuned on the IMDb dataset for binary classification.

## Steps to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Vik3927/DistilBERT-text-classification.git
cd DistilBERT-text-classification 
```

### 2. Install Dependencies
Install the required Python packages listed in the requirements.txt file:

``` bash
pip install -r requirements.txt
```

### 3. Run the Project
To train and evaluate the model, simply run the main.py file:

```bash
python main.py
```

This will:

1. Load the IMDb dataset
2. Tokenize the dataset
3. Load the pre-trained DistilBERT model
4. Fine-tune the model on the dataset
5. Evaluate the model's performance
6. Save the trained model and tokenizer


### 4. View Evaluation Results
After training, the model will be evaluated on the test set. The evaluation metrics, such as accuracy and loss, will be printed to the console.

### 5. Save and Export the Model
The trained model and tokenizer will be saved in the distilbert-imdb/ directory. You can load these later for inference or further training.

### Key Files
* main.py: The main entry point of the project. It handles the overall workflow: loading data, tokenizing, training, evaluating, and saving the model.
* tokenizer.py: Contains the tokenizer logic using Hugging Face's transformers library.
* model.py: Loads the pre-trained DistilBERT model for sequence classification.
* train.py: Contains the training loop using the Hugging Face Trainer API.
* evaluate.py: Evaluates the trained model and outputs performance metrics.
* save_model.py: Saves the model and tokenizer to disk for later use.

### Requirements
* Python 3.7+
* transformers (Hugging Face library)
* datasets (Hugging Face library)
* torch (PyTorch)

All dependencies are listed in the requirements.txt file.

### How to Modify
If you want to use this project for a different text classification task:

* Replace the dataset loading part in `main.py` with your desired dataset.
* Adjust the number of labels (e.g., for multi-class classification) in `model.py`.

### Future Work
* Extend this project to perform multi-class classification using datasets like AG News or Yelp Reviews.
* Experiment with other transformer models such as BERT or RoBERTa.
* Implement techniques to further improve model performance, such as learning rate scheduling or advanced data augmentation.

### Contributing
Feel free to contribute to this project by submitting a pull request or reporting any issues.