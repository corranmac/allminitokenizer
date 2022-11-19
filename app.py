from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    sentences = model_inputs.get('prompts', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Return the results as a dictionary
    return sentence_embeddings
