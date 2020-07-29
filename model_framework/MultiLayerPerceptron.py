import logging
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, StringProcessing, IBDOCFeaturizer, PUNC_TABLE


class FeatureEngineeringMultiLayerPerceptron(FeatureEngineering):
    def __init__(self, data_args):
        self.task = data_args['task']
        self.data = data_args['dataset']
        self.data_config = data_args['data_config']
        self.candidates_fields = data_args['candidates_fields']

    def create_train_test_data(self):
        # Balance samples by removing some non-entity labeled datapoints
        samples, targets, warnings = self.data.generate_spatial_samples(self.candidates_fields['org'], self.data_config)
        pos_idx = np.where(targets == 1)[0]
        num_pos_samples = len(pos_idx)

        neg_idxs_all = np.where(targets == 0)[0]
        np.random.shuffle(neg_idxs_all)
        neg_idx = neg_idxs_all[:num_pos_samples]

        idx_to_use = np.concatenate((pos_idx, neg_idx))

        filtered_samples = samples[idx_to_use]
        filtered_targets = targets[idx_to_use]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(filtered_samples, filtered_targets, test_size=0.3, random_state=0)
        return (X_train, X_test, y_train, y_test)


class MultiLayerPerceptron(ModelTrainer):
    def __init__(self, data_args,  training_args, model):
        self.data_args = data_args
        self.training_args = training_args
        self.model = model

    def train(self, X_train, X_test, y_train, y_test):
        logging.info('Training multilayer perceptron model for {} samples'.format(X_train.shape[0]))
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), 
            epochs=self.training_args['epochs'], batch_size=self.training_args['batch_size'])
        self.history = history

    def evaluate(self):
        logging.info(self.history.history.keys())
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss']) 
        plt.title('Model loss') 
        plt.ylabel('Loss') 
        plt.xlabel('Epoch') 
        plt.legend(['Train', 'Test'], loc='upper left') 
        plt.show()
        return self.history.history['val_accuracy'][-1]

    def analyze(self, found_mapping, comparison_fns, fields=None):
        # Sample usage: omf_paystubs.evaluate({k: {self.data_args['candidates_fields']['org']: found_companies[k]} for k in found_companies}, {}, fields=[self.data_args['candidates_fields']['org']])

        """
        comparison_fns: Map<Text, (sample, field, expected, actual, result_dict): None>
        """
        results = {
            'true_positives': [],
            'false_positives': [],
            'true_negatives': [],
            'false_negatives': []
        }

        golden = self.data_args['dataset'].golden
        to_compare = golden.columns
        if fields:
            to_compare = fields

        for sample in golden.index:
            actual = found_mapping.get(sample, {})
            for field in to_compare:
                expected_field = golden.at[sample, field]
                actual_field = actual.get(field)
                if field in comparison_fns:
                    comparison_fns[field](sample, field, expected_field, actual_field,
                                            results)
                else:
                    StringProcessing.StrictEquality()(sample, field, expected_field, actual_field,
                                results)
        return results

    def predict(self, threshold=0.60, distance_threshold=1.5):
        
        results = {}
        for dataset_file in list(self.data_args['dataset'].dataset.keys()):
            logging.info("Running predictions for file: {}".format(dataset_file))
            try:
                ibdoc = self.data_args['dataset'].dataset[dataset_file].get_joined_page()[0] # 20, 54, 70
                featurizer = IBDOCFeaturizer(ibdoc)
                fvs = featurizer.get_feature_vectors(self.data_args['data_config'])
        #         print(ibdoc.get_text())
        #         print('=================================')
                predictions = self.model.predict(fvs)
                predictions = predictions.tolist()
                sequences = [[]]
                for i, classification in enumerate(predictions):
                    if classification[0] > threshold:
                        token_to_add = featurizer.get_all_tokens()[i]
                        to_add_start, to_add_height = token_to_add['start_x'], token_to_add['line_height']
                        if len(sequences[-1]) > 0 and (to_add_start - sequences[-1][-1]['end_x']) <= distance_threshold * to_add_height:
                            sequences[-1].append(token_to_add)
                        else:
                            sequences.append([token_to_add])
                    elif len(sequences[-1]) > 0:
                        sequences.append([])
                companies = [' '.join([ss['word'] for ss in s]) for s in sequences if len(s) > 1]
                results[dataset_file] = companies
            except Exception as e:
                print(e)
        
        return results

    def analyze_result(self, found_companies):
        total = 0
        found_count = 0
        for cfile in found_companies:
            try:
                expected = self.data_args['dataset'].golden.at[cfile, self.data_args['candidates_fields']['org']]
            except Exception as e:
                print(e)
                continue
            found = '\n\t\t'.join(found_companies[cfile])
            print(cfile[-10:])
            print('\t Found:\n\t\t{}'.format(found))
            print('\t Expected:\n\t\t{}'.format(expected))
            if expected:  
                total += 1
            expected_san = expected.lower().strip().translate(PUNC_TABLE)
            actual_san = [c.lower().strip().translate(PUNC_TABLE) for c in found_companies[cfile]]
            is_contained = any([(expected_san in a) for a in actual_san]) or any([(a in expected_san) for a in actual_san])
            if expected_san in actual_san or is_contained:
                found_count += 1
            else:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(total)
        print(found_count)
        print('Recall: {}'.format(float(found_count)/float(total)))

    def demo(self):
        ibdoc = self.data_args['dataset'].dataset[list(self.data_args['dataset'].dataset.keys())[54]].get_joined_page()[0] # 20, 54, 70
        featurizer = IBDOCFeaturizer(ibdoc)
        fvs = featurizer.get_feature_vectors(self.data_args['data_config'])
        print(ibdoc.get_text())
        print('=================================')
        predictions = self.model.predict(fvs)
        predictions = predictions.tolist()
        sequences = [[]]
        for i, classification in enumerate(predictions):
            if classification[0] > 0.99:
                token_to_add = featurizer.get_all_tokens()[i]
                to_add_start, to_add_height = token_to_add['start_x'], token_to_add['line_height']
                if len(sequences[-1]) > 0 and (to_add_start - sequences[-1][-1]['end_x']) <= 1.5 * to_add_height:
                    sequences[-1].append(token_to_add)
                else:
                    sequences.append([token_to_add])
            elif len(sequences[-1]) > 0:
                sequences.append([])
        companies = [' '.join([ss['word'] for ss in s]) for s in sequences if len(s) > 1]
        print(companies)
