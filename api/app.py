#!/usr/bin/env python
from flask import Flask, request, jsonify
import os
import torch
import logging
from transformers import CamembertForSequenceClassification, CamembertTokenizer


logging.basicConfig(format='%(message)s', level=logging.INFO)


app = Flask(__name__)
app.config["DEBUG"] = False


def get_model():
    """ Load a trained model.

    Returns:
        CamembertForSequenceClassification: model class
    """
    state_dict = torch.load("lingorank.pt", map_location=torch.device('cpu'))
    model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        state_dict=state_dict,
        num_labels=6)
    return model


def preprocess(raw_texts, labels=None):
    """ Takes raw data as argument and returns a pytorch dataloader.

    Args:
        raw_texts (array-like) : A list of texts in the form of 'str'
        labels : a labels list from 0 to 5

    Returns:
        inputs_ids, attention_masks, labels(optionel) : PyTorch object that contains tokenized and encoded versions of raw data
    """
    TOKENIZER = CamembertTokenizer.from_pretrained(
        'camembert-base', do_lower_case=True)
    encoded_batch = TOKENIZER.batch_encode_plus(raw_texts,
                                                add_special_tokens=True,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
    if labels:
        labels = torch.tensor(labels)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], labels
    return encoded_batch['input_ids'], encoded_batch['attention_mask']


def predict(texts, model):
    """ Do predictions.

    Args:
        texts (array-like): a list of texts
        model (CamembertForSequenceClassification): trained model

    Returns:
        predictions
    """
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask = preprocess(texts)
        retour = model(input_ids, attention_mask=attention_mask)
        return torch.argmax(retour[0], dim=1)


model = get_model()
difficulties = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}


@app.route('/', methods=['GET'])
def home():
    return "BSA2021 - Team Google's API is working!"


@app.route('/api/predict', methods=['GET'])
def make_prediction():
    text = request.args.get("text", "")
    prediction = predict([text], model)
    return jsonify({'text': text, 'difficulty': difficulties[int(prediction)]}), 200


@app.route('/api/predict/words', methods=['GET'])
def make_prediction_words():
    text = request.args.get("text", "")
    predictions = predict(text.split(' '), model)
    words = []
    for prediction in predictions:
        words.append(int(prediction))
    return jsonify({'text': text, 'difficulty': words}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
