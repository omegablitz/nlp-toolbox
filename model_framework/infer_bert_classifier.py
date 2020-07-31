import argparse
import bert_utils
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os, sys
from bert_utils import convert_sentences_to_bert_inputs, flatten_tensor_and_get_accuracy, save_plots_models
from transformers import BertForSequenceClassification, BertTokenizer
import logging
import transformers

sys.path.insert(0, os.path.abspath('..'))
os.environ["PYTHONIOENCODING"] = "utf-8"

def get_classifier_inference(model_type, modelPath, testdf, num_labels, gpu):
    logging.getLogger('transformers').setLevel(logging.WARNING)

    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    logging.debug('Device: {}'.format(device))

    model_state_dict = torch.load(modelPath, map_location=device)
    model = BertForSequenceClassification.from_pretrained(model_type, state_dict=model_state_dict, num_labels=num_labels)
    logging.debug("Fine-tuned model loaded with labels = {}".format(model.num_labels))

    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)

    # testing
    input_ids, labels, attention_masks = convert_sentences_to_bert_inputs(tokenizer, testdf)
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = 32
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()

    # Tracking variables
    predictions, true_labels = [], []
    nb_eval_steps = 0
    eval_accuracy = 0
    csv_output = []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        # shape (batch_size, config.num_labels)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        tmp_eval_accuracy = flatten_tensor_and_get_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        count = 0
        for i in range(pred_flat.shape[0]):
            # iterate over the batch
            csv_output.append((b_input_ids[i], pred_flat[i], labels_flat[i]))

    validLabels = [x for x in labels if str(x) != 'nan']

    if len(validLabels) != 0:
        # otherwise it is inference mode with all candidate phrases (with label = NA)
        logging.info('Test Accuracy Accuracy: {0:0.4f}'.format((float(eval_accuracy) / float(nb_eval_steps))))

        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        labels = [x for x in range(num_labels - 1)]    # don't include 'None' class

        micro_precision = precision_score(flat_true_labels, flat_predictions, labels=labels, average="micro")
        micro_recall = recall_score(flat_true_labels, flat_predictions,  labels=labels, average="micro")
        micro_f1 = f1_score(flat_true_labels, flat_predictions, labels=labels, average="micro")
        logging.info('Micro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(micro_recall, micro_precision, micro_f1))

        macro_precision = precision_score(flat_true_labels, flat_predictions, labels=labels, average="macro")
        macro_recall = recall_score(flat_true_labels, flat_predictions, labels=labels, average="macro")
        macro_f1 = f1_score(flat_true_labels, flat_predictions, labels=labels, average="macro")
        logging.info('Macro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(macro_recall, macro_precision, macro_f1))

    sentences = testdf['context'].tolist()  # un-tokenized sentences

    assert (len(sentences) == len(csv_output))

    headings = ['context', 'predicted', 'label']
    df = pd.DataFrame(columns=headings)
    index = 0
    for ids, pred, label in csv_output:
        ids = np.trim_zeros(ids.cpu().numpy())
        sentence = tokenizer.convert_ids_to_tokens(ids)[1:-1]
        # data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
        data = [{'context': sentences[index], 'predicted': str(pred), 'label': str(label)}]
        df = df.append(pd.DataFrame(data, columns=headings))
        index += 1


    return df