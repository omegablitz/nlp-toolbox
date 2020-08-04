import json
import ast
import logging
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, StringProcessing


class TaskNER(Task):
    def __init__(self, config):
        self.config = config
        self.set_labels_dict()

    def set_labels_dict(self):
        self.labels_dict = self.config['labels_dict']
        self.cat_dict = {v: k for k, v in self.labels_dict.items()}


class Refiner(ModelTrainer):
    def __init__(self, data_args, training_args, models_to_evaluate):
        self.data_args = data_args
        self.training_args = training_args
        self.models_to_evaluate = models_to_evaluate

    def analyze_results(self):
        result_file_path = self.training_args['model_file_or_path']
        models = self.models_to_evaluate['models']
        spacy_models = self.models_to_evaluate['spacy_models']
        person_name_models = self.models_to_evaluate['person_name_models']
        org_name_models = self.models_to_evaluate['org_name_models']

        all_recall = {}
        all_precision = {}

        final_results = {}
        final_results['person'] = {}
        final_results['org'] = {}

        f = open(result_file_path)
        results = json.load(f)
        for result in results:
            # one file at a time
            key = result['absolute_ibocr_path'].split('/')[-1]
            key = ".".join(key.split('.')[:-1])
            names = {}
            phrases = result['refined_phrases']
            for phrase in phrases:
                # one model/refiner field at a time
                label = phrase['label']
                if label in models:
                    word_list = ast.literal_eval(phrase["word"])
                    cleaned_word_list = [" ".join(w.split()) for w in word_list]

                elif label in spacy_models:
                    jsonstr = phrase["word"]
                    json_dict = json.loads(jsonstr)
                    entities = json_dict["entities"]
                    cleaned_word_list = [" ".join(ent_json['entity'].split()) for ent_json in entities]
                    # remove all occurences of empty string
                    cleaned_word_list = list(filter(('').__ne__, cleaned_word_list))


                names[label] = cleaned_word_list

            relevant_person = self.data_args['dataset'].golden.loc[key, self.data_args['candidates_fields']['person']]
            relevant_org = self.data_args['dataset'].golden.loc[key, self.data_args['candidates_fields']['org']]

            # PERSON NAMES
            for model in person_name_models:
                retrieved, relevant = Evaluation.get_Retr_Rel_Set(names[model], relevant_person)
                precision, recall = Evaluation.get_Precision_Recall(retrieved, relevant)

                if model in final_results['person']:
                    final_results['person'][model][key] = retrieved
                else:
                    final_results['person'][model] = {}

                if model in all_recall:
                    all_recall[model].append(recall)
                    all_precision[model].append(precision)
                else:
                    all_recall[model] = [recall]
                    all_precision[model] = [precision]


            # ORG NAMES
            for model in org_name_models:
                retrieved, relevant = Evaluation.get_Retr_Rel_Set(names[model], relevant_org)
                precision, recall = Evaluation.get_Precision_Recall(retrieved, relevant)

                if model in final_results['org']:
                    final_results['org'][model][key] = retrieved
                else:
                    final_results['org'][model] = {}

                if model in all_recall:
                    all_recall[model].append(recall)
                    all_precision[model].append(precision)
                else:
                    all_recall[model] = [recall]
                    all_precision[model] = [precision]

        Evaluation.print_scores(all_recall, all_precision, person_name_models, org_name_models)
        return final_results

    def demo(self, results, filename):
        # Print results
        for typ in results:
            logging.info('Field type: {}'.format(typ))
            for model in results[typ]:
                logging.info('model type: {}'.format(model))
                logging.info(results[typ][model][filename])
                logging.info("\n")
            logging.info("\n")
