# colab-sft-project

## Overview
This project demonstrates the process of Supervised Fine-Tuning (SFT) for language models using PyTorch and the Hugging Face Transformers library. The main component of the project is a Jupyter notebook that guides users through the steps of loading models, testing them with sample questions, and performing SFT on a smaller model.

## Project Structure
- **SFT_colab_demo.ipynb**: The main Jupyter notebook that includes sections for installing necessary libraries, importing libraries, loading models and tokenizers, testing the models with sample questions, and performing supervised fine-tuning (SFT) on a small model.
- **utils.py**: A Python file containing helper functions used in the notebook, such as `generate_responses`, `test_model_with_questions`, `load_model_and_tokenizer`, and `display_dataset`. These functions facilitate model loading, response generation, and dataset display.
- **requirements.txt**: A text file listing the necessary Python packages required for the project, including `torch`, `transformers`, `datasets`, and `trl`. This ensures that all dependencies are installed for the notebook to run smoothly.

## Setup Instructions
1. **Clone the Repository**: 
   Clone this repository to your local machine or Google Colab environment.

2. **Install Dependencies**: 
   Run the following command to install the required libraries:
   ```
   !pip install -r requirements.txt
   ```

3. **Open the Notebook**: 
   Open `SFT_colab_demo.ipynb` in Jupyter Notebook or Google Colab.

4. **Run the Notebook**: 
   Follow the instructions in the notebook to load models, test them with questions, and perform supervised fine-tuning.

## Functionality
- **Model Loading**: Load pre-trained models and tokenizers for testing and fine-tuning.
- **Testing with Questions**: Evaluate the models' responses to predefined questions to assess their performance.
- **Supervised Fine-Tuning**: Fine-tune a smaller model on a specific dataset to improve its performance on targeted tasks.

## Notes
- Ensure that you have access to a GPU if you plan to run the fine-tuning process on larger models.
- The notebook is designed to be user-friendly, with clear instructions and outputs for each step of the process.