# Conversational Chatbot using GPT-2

This project utilizes the GPT-2 language model to build a conversational chatbot. The chatbot is trained on movie dialogue data from the Cornell Movie-Dialogs Corpus.

## Files Included

1. `train_chatbot.py`: This Python script contains the code for training the chatbot model.
2. `test_chatbot.py`: This Python script allows interactive testing of the trained chatbot model.

## Dependencies

- Python 3.x
- TensorFlow
- Transformers library by Hugging Face
- Cornell Movie-Dialogs Corpus (included in the repository)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/your_repository.git
    ```

2. Install the required dependencies:

    ```bash
    pip install tensorflow transformers
    ```

## Usage

1. **Training the Chatbot Model:**

    Run the `train_chatbot.py` script to train the chatbot model. This script preprocesses the movie dialogue data, tokenizes it, pads the sequences, compiles the model, and trains it. The trained model will be saved in a directory named `chatbot_model`.

    ```bash
    python train_chatbot.py
    ```

2. **Testing the Chatbot:**

    After training the model, you can interactively test it using the `test_chatbot.py` script. This script prompts the user to input text, generates a response from the trained chatbot model, and prints the response.

    ```bash
    python test_chatbot.py
    ```

    Enter your text when prompted with `User:` and observe the response from the chatbot.

## Notes

- The training process may take some time depending on the size of the dataset and the hardware used.
- You can adjust the hyperparameters such as batch size, learning rate, and number of epochs in the `train_chatbot.py` script according to your requirements.
- Ensure that you have sufficient disk space available, especially if you are training on a large dataset.
