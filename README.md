# D&D AI Chat Application

## Project Overview
This project is an interactive chat application that leverages a TensorFlow model to interact with users using sentences from Dungeons & Dragons sessions. The application loads sentences from a CSV file, trains a neural network on this data, and engages the user in dialogue, responding based on user prompts.

## Features
- **TensorFlow Neural Network**: Utilizes a simple neural network to process and respond to user inputs.
- **Interactive Chat**: Engages users with dynamic responses generated from a dataset of D&D sentences.
- **Data Handling**: Manages data from `Extended_Sentences.csv`, providing content for AI responses.

## Installation

### Prerequisites
- Python 3.6 or newer.
- `pip` for installing Python packages.

### Dependencies
Install all necessary Python packages using:

```bash
pip install -r requirements.txt
```




## Files
    app.py: Main script with TensorFlow model and chat functionality.
    Extended_Sentences.csv: CSV file with sentences for AI interactions.
    requirements.txt: Lists all required Python libraries.
### Usage
    To run the application:

    ```bash
    Copy code
    python app.py
    ```
    This starts the interactive chat where the AI begins with a random sentence from the CSV. Users can respond to prompts, and the AI will generate responses based on its neural network training.

### How It Works
    Sentence Loading: Loads sentences from Extended_Sentences.csv to form the basis of AI interactions.
    Model Training: Trains a model on these sentences to understand context and generate responses. (Note: In a production environment, consider training the model separately and loading a trained model for interaction.)
    User Interaction: Users interact with the AI through prompts, and the AI responds based on learned patterns.
    Exiting the Chat
    To exit the application, type 'exit' during any interaction prompt.