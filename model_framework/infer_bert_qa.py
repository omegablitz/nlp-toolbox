import os
import torch
import time
import pandas as pd
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# Config
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0
logging.getLogger('transformers').setLevel(logging.WARNING)

def to_list(tensor):
        return tensor.detach().cpu().tolist()

def run_prediction(model, tokenizer, device, output_dir, filename, question_texts, context_text):
        """Setup function to compute predictions"""
        examples = []

        for i, question_text in enumerate(question_texts):
            example = SquadExample(
                qas_id=str(i),
                question_text=question_text,
                context_text=context_text,
                answer_text=None,
                start_position_character=None,
                title="Predict",
                is_impossible=False,
                answers=None,
            )

            examples.append(example)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

        all_results = []

        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                example_indices = batch[3]

                outputs = model(**inputs)

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)

                    output = [to_list(output[i]) for output in outputs]

                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)
                    all_results.append(result)

        output_prediction_file = os.path.join(output_dir, filename + "_predictions.json")
        output_nbest_file = os.path.join(output_dir, filename + "_nbest_predictions.json")
        output_null_log_odds_file = os.path.join(output_dir, filename + "_null_predictions.json")

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            False,  # verbose_logging
            True,  # version_2_with_negative
            null_score_diff_threshold,
            tokenizer,
        )

        return predictions


def get_qa_inference(data_texts, queries, model_name_or_path, output_dir, gpu):
    # Setup model
    config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    model = model_class.from_pretrained(model_name_or_path, config=config)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # device = torch.device("cpu")

    model.to(device)

    
    logging.info(' Total number of Files: {}'.format(len(data_texts.keys())))
    
    
    headers = ['filename']
    for query in queries:
        headers.append(query)
    results = pd.DataFrame(columns=headers)
    
    for key in data_texts:
        logging.info('File name: {}'.format(key))
        context = data_texts[key]
        # Run method
        predictions = run_prediction(model, tokenizer, device, output_dir, key, queries, context)

        this_results = [key]
        # Print results
        for ind, key in enumerate(predictions.keys()):
            print(queries[ind] + " : " + predictions[key])
            this_results.append(predictions[key])
        
        
        a_series = pd.Series(this_results, index = results.columns)
        results = results.append(a_series, ignore_index=True)

    return results
