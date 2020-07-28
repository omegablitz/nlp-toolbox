import pandas as pd
import logging
import numpy as np
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation
from infer_bert_classifier import get_classifier_inference
from sklearn.metrics import f1_score, precision_score, recall_score


class TaskNER(Task):
    def __init__(self, config):
        self.config = config
        self.set_labels_dict()

    def set_labels_dict(self):
        self.labels_dict = self.config['labels_dict']
        self.cat_dict = {v: k for k, v in self.labels_dict.items()}


class FeatureEngineeringNER(FeatureEngineering):
    def __init__(self, data_args):
        self.labels_dict = data_args['task'].labels_dict
        self.data = data_args['dataset']
        self.candidates_fields = data_args['candidates_fields']

    def generate_test_samples_from_goldens(self):
        labels = []
        contexts = []

        persons = self.data.golden[self.candidates_fields['person']].tolist()
        person_label = [self.labels_dict['person']] * len(persons)

        contexts.extend(persons)
        labels.extend(person_label)

        org = self.data.golden[self.candidates_fields['org']].tolist()
        org_label = [self.labels_dict['org']] * len(org)

        contexts.extend(org)
        labels.extend(org_label)

        test_samples = pd.DataFrame(list(zip(contexts, labels)), columns=['context', 'label'])
        test_samples = test_samples.sample(frac=1)
        test_samples = test_samples.dropna()
        return test_samples

    def generate_test_samples_from_candidates(self):
        test_samples = {}
        for key in self.data.candidates:
            candidate_list = self.data.candidates[key]
            labels = [np.nan] * len(candidate_list)
            df = pd.DataFrame(list(zip(candidate_list, labels)), columns=['context', 'label'])
            df = df.drop_duplicates()  # redundant strings
            df['context'] = df['context'].astype(str)
            df['label'] = df['label'].astype(float)
            test_samples[key] = df
        return test_samples


class BERTNER(ModelTrainer):
    def train(self):
        pass

    def evaluate(self):
        # cross validation accuracy
        pass

    def predict(self, testdata):
        self.labels_dict = self.data_args['task'].labels_dict
        self.cat_dict = self.data_args['task'].cat_dict
        self.num_labels = self.training_args['num_labels']

        model_file_or_path = self.training_args['model_file_or_path']
        
        gpu = self.training_args['gpu']
        if isinstance(testdata, pd.DataFrame): 
            # testdata is single dataframe as data is generated using goldens csv
            logging.info("inferring BERT classifier for single df generated from goldens csv of size {}".format(testdata.shape))
            return get_classifier_inference(model_file_or_path, testdata, self.num_labels, gpu)
        elif isinstance(testdata, dict):
            # test_data is a dictionary {'filename' : dataframe}
            results = {}
            for key in testdata:
                logging.info("inferring BERT classifier for file {}".format(key))
                results[key] = get_classifier_inference(model_file_or_path, testdata[key], self.num_labels, gpu)
            
            return results
    
    def analyze_golden_result(self, results):
        labels = [x for x in range(self.num_labels - 1)] # Not to include 'None' class since it is never in true_label
        true_labels = results['label']
        predictions = results['predicted']

        # Overall Score
        micro_precision = precision_score(true_labels, predictions, labels=labels, average="micro")
        micro_recall = recall_score(true_labels, predictions,  labels=labels, average="micro")
        micro_f1 = f1_score(true_labels, predictions, labels=labels, average="micro")
        logging.info('Micro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(micro_recall, micro_precision, micro_f1))

        macro_precision = precision_score(true_labels, predictions, labels=labels, average="macro")
        macro_recall = recall_score(true_labels, predictions, labels=labels, average="macro")
        macro_f1 = f1_score(true_labels, predictions, labels=labels, average="macro")
        logging.info('Macro Test R: {0:0.4f}, P: {1:0.4f}, F1: {2:0.4f}'.format(macro_recall, macro_precision, macro_f1))

        # Class Wise Score
        for lab in labels:
            # micro vs macro doesn't matter in case of single-label
            precision = precision_score(true_labels, predictions, labels=[lab], average="micro")
            recall = recall_score(true_labels, predictions,  labels=[lab], average="micro")
            f1 = f1_score(true_labels, predictions, labels=[lab], average="micro")
            logging.info('Category: {0}, Test R: {1:0.4f}, P: {2:0.4f}, F1: {3:0.4f}'.format(self.cat_dict[lab], recall, precision, f1))
    
    def analyze_result(self, results):
        # results is a list of dataframes (one for each file) -- BERT's outputs
        person_recall = []
        person_precision = []
        org_recall = []
        org_precision = []

        final_results = {}
        final_results['person'] = {}
        final_results['org'] = {}

        for key in results:
            bert_results = results[key]
            bert_results['predicted'] = bert_results['predicted'].astype(int)
            bert_person_df = bert_results[bert_results['predicted'] == self.labels_dict['person']]
            bert_org_df = bert_results[bert_results['predicted'] == self.labels_dict['org']]
            
            retrieved_person = bert_person_df['context'].tolist()
            retrieved_org = bert_org_df['context'].tolist()
            relevant_person = self.data_args['dataset'].golden.loc[key, self.data_args['candidates_fields']['person']]
            relevant_org = self.data_args['dataset'].golden.loc[key, self.data_args['candidates_fields']['org']]
            
            # PERSON NAMES
            retrieved, relevant = Evaluation.get_Retr_Rel_Set(retrieved_person, relevant_person)
            precision, recall = Evaluation.get_Precision_Recall(retrieved, relevant)
            person_precision.append(precision)
            person_recall.append(recall)
            final_results['person'][key] = retrieved

            # ORG NAMES
            retrieved, relevant = Evaluation.get_Retr_Rel_Set(retrieved_org, relevant_org)
            precision, recall = Evaluation.get_Precision_Recall(retrieved, relevant)
            org_precision.append(precision)
            org_recall.append(recall)
            final_results['org'][key] = retrieved
            

        r = np.mean(person_recall)
        p = np.mean(person_precision)
        f1 = 2*p*r/(p + r)
        logging.info("For field {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format('person', r, p, f1))

        r = np.mean(org_recall)
        p = np.mean(org_precision)
        f1 = 2*p*r/(p + r)
        logging.info("For field {0}, recall: {1:0.4f}, precision: {2:0.4f}, F1: {3:0.4f} ".format('org', r, p, f1))
        return final_results

    def demo(self, results):
        # Print results
        for typ in results:
            logging.info('Field type: {}'.format(typ))
            for key in results[typ]:
                logging.info('filename: {}'.format(key))
                logging.info(results[typ][key])

