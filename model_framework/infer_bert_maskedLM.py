import os
import torch
import time
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    squad_convert_examples_to_features
)


def get_lm_inference(queries, model_name_or_path, output_dir, gpu):
    # Setup model
    config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    model = model_class.from_pretrained(model_name_or_path, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    data_dir = "/Users/ahsaasbajaj/Documents/Data/CARTA/processed_texts/"
    files = os.listdir(data_dir)
    print('Files: ', len(files))

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        print('File name: ', fname)
        f = open(fpath)
        context = f.read()

        # only for debugging
        context = context[:5000]
        
        # Run method
        predictions = run_prediction(model, tokenizer, device, output_dir, queries, context)

        outpath = os.path.join(output_dir, fname)
        outf = open(outpath, 'w')

        # Print results
        for ind, key in enumerate(predictions.keys()):
            print(queries[ind] + " : " + predictions[key])
            print(queries[ind] + " : " + predictions[key], file=outf)

        outf.close()