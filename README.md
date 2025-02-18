# FacebookAI/roberta-base Fine-Tuned Model for Mask Filling

This repository hosts a fine-tuned version of the **FacebookAI/roberta-base** model, optimized for **mask filling** tasks using the **Salesforce/wikitext** dataset. The model is designed to perform fill-mask operations efficiently while maintaining high accuracy.

## Model Details
- **Model Architecture:** RoBERTa  
- **Task:** Mask Filling  
- **Dataset:** Hugging Face's ‘Salesforce/wikitext’ (wikitext-2-raw-v1)
- **Quantization:** FP16
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage
### Installation
```sh
from transformers import RobertaForMaskedLM, RobertaTokenizer
import torch

# Load the fine-tuned RoBERTa model and tokenizer
model_name = 'roberta_finetuned'  # Your fine-tuned RoBERTa model
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Quantize the model to FP16
model = model.half()

# Save the quantized model and tokenizer
model.save_pretrained("./quantized_roberta_model")
tokenizer.save_pretrained("./quantized_roberta_model")

# Example input for testing (10 sentences)
input_texts = [
    "The sky is <mask> during the night.",
    "Machine learning is a subset of <mask> intelligence.",
    "The largest planet in the solar system is <mask>.",
    "The Eiffel Tower is located in <mask>.",
    "The sun rises in the <mask>.",
    "Mount Everest is the highest mountain in the <mask>.",
    "The capital of Japan is <mask>.",
    "Shakespeare wrote Romeo and <mask>.",
    "The currency of the United States is <mask>.",
    "The fastest land animal is the <mask>."
]

# Process each input sentence
for input_text in input_texts:
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the prediction for the masked token
    masked_index = inputs.input_ids[0].tolist().index(tokenizer.mask_token_id)
    predicted_token_id = logits[0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)

    print(f"Input: {input_text}")
    print(f"Predicted token: {predicted_token}\n")
```
📊 Evaluation Results
After fine-tuning the RoBERTa-base model for mask filling, we evaluated the model's performance on the validation set from the Salesforce/wikitext dataset. The following results were obtained:

Metric	Score	Meaning
Bleu Score: 0.8

## Fine-Tuning Details
### Dataset
The Hugging Face's `medical-qa-datasets’ dataset was used, containing different types of Patient and Doctor Questions and respective Answers.

### Training
- **Number of epochs:** 3  
- **Batch size:** 8  
- **Evaluation strategy:** steps

### Quantization
Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure
```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

Limitations
The model is primarily trained on the wikitext-2 dataset and may not perform well on highly domain-specific text without additional fine-tuning.
The model may not handle edge cases involving unusual grammar or rare words as effectively.
Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.


