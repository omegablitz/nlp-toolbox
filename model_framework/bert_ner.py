import pandas as pd
import logging
import random
import numpy as np
import re
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, DataCuration
from infer_bert_classifier import get_classifier_inference
from bert_finetuning import finetune_bert
from sklearn.metrics import f1_score, precision_score, recall_score
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

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
        self.data_args = data_args

    @staticmethod
    def create_train_test_data_from_files(data_args):
        # add support to read from instabase
        labels_dict = data_args['task'].labels_dict

        cmp = pd.read_csv(data_args['filepaths']['org'])
        cmps = cmp[data_args['name_field']['org']].to_list()
        logging.info("Total {0} company names available".format(len(cmps)))

        fnames = pd.read_csv(data_args['filepaths']['person1'])
        lnames = pd.read_csv(data_args['filepaths']['person2'])

        first = fnames[data_args['name_field']['person']].to_list() 
        last = lnames[data_args['name_field']['person']].to_list()

        logging.info("Total {0} first names, {1} last names available".format(len(first), len(last)))


        df = pd.read_csv(data_args['filepaths']['none'])
        # similar to ones generated by def get_candidates_filtered_by_names(self, data_split, all_names)

        none_phrases = df[data_args['name_field']['none']].tolist()
        logging.info("Total {0} none phrases available".format(len(none_phrases))) 

        NUM_SAMPLES = min(len(cmps), len(first), len(last), len(none_phrases))
        logging.info("Selecting minimum of three categories available, min count: {0}".format(NUM_SAMPLES))
        cmp_sample = random.sample(cmps, NUM_SAMPLES)
        first_sample = random.sample(first, NUM_SAMPLES)
        last_sample = random.sample(last, NUM_SAMPLES)
        none_phrases_sample = random.sample(none_phrases, NUM_SAMPLES)

        names_sample = []
        for f, l in zip(first_sample, last_sample):
            if isinstance(f, str) and isinstance(l, str):
                names_sample.append(f + " " + l)


        all_names = []
        all_labels = []

        all_names.extend(names_sample)
        all_labels.extend([labels_dict['person']] * len(names_sample))

        all_names.extend(cmp_sample)
        all_labels.extend([labels_dict['org']] * len(cmp_sample))

        all_names.extend(none_phrases_sample)
        all_labels.extend([labels_dict['none']] * len(none_phrases_sample))

        data = pd.DataFrame(zip(all_names, all_labels), columns=['context', 'label'])
        data = data.sample(frac=1)

        train_data, test_data = DataCuration._split_train_test(data, per=0.7)
        return train_data, test_data

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


    def get_candidates_filtered_by_names(self, data_split, all_names):
        none_phrases = []
        for key in self.data.candidates:
            # every key is a new document, candidates has whole document
            if key in data_split.index:
                # if present in particular split (train, test)
                this_candidates = self.data.candidates[key]
                this_names = all_names[key]
                filtered_candidates = []
                for cds in this_candidates:
                    if cds not in this_names:
                        filtered_candidates.append(cds)

                none_phrases.extend(this_candidates)

        return none_phrases

    def generate_labeled_data(self, data_split, processing_config):
        logging.info('Generating labeled data for dataset with shape {0}'.format(data_split.shape))

        all_texts = {}
        for key in data_split.index:
            text = self.data.texts[key]
            cleaned_text = re.sub('\s+', ' ', text)
            all_texts[key] = cleaned_text
        
        LEN = len(all_texts.keys())
        logging.info("Cleaned text size {0}".format(LEN))

        # Series in golden dataframe converted to lists
        filenames = data_split.index.tolist()
        employee_name = data_split[self.data_args['candidates_fields']['person']].tolist()
        employer_name = data_split[self.data_args['candidates_fields']['org']].tolist()
        employer_address = (data_split[self.data_args['candidates_fields']['address1'][0]] + ' ' + data_split[self.data_args['candidates_fields']['address1'][1]]).tolist()
        employee_address = (data_split[self.data_args['candidates_fields']['address2'][0]] + ' ' + data_split[self.data_args['candidates_fields']['address2'][1]]).tolist()
        # use as negative filter
        other_location = (data_split[self.data_args['candidates_fields']['address3'][0]] + ' ' + data_split[self.data_args['candidates_fields']['address3'][1]] + ' ' + 
            data_split[self.data_args['candidates_fields']['address3'][2]] + ' ' + data_split[self.data_args['candidates_fields']['address3'][3]]).tolist()

        all_stop_words = {}
        all_names = {}
        # remove person and org and addresses to get data for 'None' class
        for key, a, b, c, d, e in zip(filenames, employee_name, employer_name, employer_address, employee_address, other_location):
            stop_words = []
            stop_words.extend(a.split(' '))
            stop_words.extend(b.split(' '))
            stop_words.extend(c.split(' '))
            stop_words.extend(d.split(' '))
            stop_words.extend(e.split(' '))

            # all names together as strings
            all_names[key] = [a, b, c, d, e]
            all_stop_words[key] = stop_words
    
        nltk_stopwords = stopwords.words('english')

        if self.data_args['use_random_seq']:
            # get LEN random_sequences (one per document)
            random_sequences = self.generate_random_sequences(all_texts, all_stop_words, nltk_stopwords, processing_config)
        else:
            # use candidates generated by DataCuration().generate_candidates_phrases()
            all_random_sequences = self.get_candidates_filtered_by_names(data_split, all_names)
            random_sequences = random.sample(all_random_sequences, LEN)

        logging.info("Random Sequence text size {0}".format(len(random_sequences)))

        person_names = employee_name
        org_names = employer_name
        person_names = [re.sub('\s{1,6}', ' ', person) for person in person_names]   

        logging.info('Mode {}'.format(self.data_args['mode']))
        labels = self.data_args['task'].labels_dict
        person_labels = len(person_names)*[labels['person']]
        org_labels = len(org_names)*[labels['org']]
        random_labels = len(random_sequences)*[labels['none']]

        person_df = pd.DataFrame(list(zip(person_names, person_labels)), columns=['context', 'label'])
        org_df = pd.DataFrame(list(zip(org_names, org_labels)), columns=['context', 'label'])
        none_df = pd.DataFrame(list(zip(random_sequences, random_labels)), columns=['context', 'label'])

        list_df = [person_df, org_df, none_df]

        if self.data_args['mode'] == "person-org-address":
            address = [] # has double the number of elements
            address.extend(employer_address)
            address.extend(employee_address)
            random.shuffle(address)
            address = address[:LEN]
            address = [re.sub('\s{1,6}', ' ', add) for add in address]

            # remove street address numebrs and zip codes, else classifier might overfit on numerical values  
            address = [re.sub('\d+', '', add) for add in address]
            address = [" ".join(add.split('-')[:-1]) for add in address] # remove dangling hiphens in zip-codes (after numbers removed)
            address = [add.strip() for add in address]

            # make sure labels_dict in Task() has a label as 'address'            
            address_labels = len(address)*[labels['address']]
            address_df = pd.DataFrame(list(zip(address, address_labels)), columns=['context', 'label'])
            list_df.append(address_df)

        result = pd.DataFrame()
        for df in list_df:
            result = result.append(df)
        result = result.sample(frac=1)            

        return result


    def generate_random_sequences(self, all_texts, all_stop_words, nltk_stopwords, processing_config):
        # alternate to using self.candidates

        RANDOM_SEQ_LEN = processing_config['RANDOM_SEQ_LEN']
        random_texts = []
        # has removed person names, addresses, organizations

        for key in all_texts:
            text = all_texts[key]
            text_tokens = text.split(' ')
            stop_words = all_stop_words[key]
            stop_words.extend(nltk_stopwords) # add nltk stopwords
            out = [word for word in text_tokens if not word in stop_words]
            random_text = " ".join(e for e in out if e.isalnum())
            random_texts.append(random_text)
            
        # generate random sequences (comparable to lengths of phrases in other categories)
        # Following generates one random sequence per document
        random_sequences = []
        for text in random_texts:
            tokens = text.split(' ')
            if len(tokens) < RANDOM_SEQ_LEN:
                continue
            start_id = random.randrange(len(tokens) - RANDOM_SEQ_LEN)
            selected_tokens = tokens[start_id : start_id + RANDOM_SEQ_LEN]
            random_sequence = " ".join(selected_tokens)
            random_sequence = re.sub('\d+', '', random_sequence) # remove numerical values to prevent overfitting
            random_sequences.append(random_sequence)

        return random_sequences


    def create_train_test_data(self, processing_config):        
        train_data = self.generate_labeled_data(self.data.golden_train, processing_config)
        test_data = self.generate_labeled_data(self.data.golden_test, processing_config)

        logging.info('train data: {0}, test test: {1}'.format(train_data.shape, test_data.shape))
        return train_data, test_data


class BERTNER(ModelTrainer):
    def train(self):
        logging.info("Finetuning BERT Model")
        finetune_bert(**self.training_args)

    def predict(self, testdata):
        self.labels_dict = self.data_args['task'].labels_dict
        self.cat_dict = self.data_args['task'].cat_dict
        self.num_labels = self.training_args['num_labels']
        model_type = self.training_args['model_type']
        model_file_or_path = self.training_args['model_file_or_path']
        gpu = self.training_args['gpu']
        
        if isinstance(testdata, pd.DataFrame): 
            # testdata is single dataframe as data is generated using goldens csv
            logging.info("inferring BERT classifier for single df generated from goldens csv of size {}".format(testdata.shape))
            return get_classifier_inference(model_type, model_file_or_path, testdata, self.num_labels, gpu)
        elif isinstance(testdata, dict):
            # test_data is a dictionary {'filename' : dataframe}
            results = {}
            for key in testdata:
                logging.info("inferring BERT classifier for file {}".format(key))
                results[key] = get_classifier_inference(model_type, model_file_or_path, testdata[key], self.num_labels, gpu)
            
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

    def demo(self, results, filename):
        # Print results
        for typ in results:
            logging.info('Field type: {}'.format(typ))
            logging.info('filename: {}'.format(filename))
            logging.info(results[typ][filename])

