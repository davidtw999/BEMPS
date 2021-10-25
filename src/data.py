import logging
import os
from typing import List, Union
from transformers.data import DataProcessor, InputExample, InputFeatures
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    PreTrainedTokenizer,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors
)


## data embedding
def data_embedding(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length=None,
    label_list=None,
    output_mode=None,
):

    logger = logging.getLogger(__name__)

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length, padding=True, truncation=True, return_token_type_ids=True
    )

    features = []

    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:2]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


## data processor
class DataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AGNewsProcessor(DataProcessor):
    def get_labels(self):
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        return labels

class IMDBProcessor(DataProcessor):
    def get_labels(self):
        labels = ["pos", "neg"]
        return labels

class PubMedProcessor(DataProcessor):
    def get_labels(self):
        labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
        return labels


class Sst5Processor(DataProcessor):
    def get_labels(self):
        labels = ["__label__1", "__label__2", "__label__3", "__label__4", "__label__5"]
        return labels


def get_processor():
    processors = glue_processors.copy()
    processors.update(
        {"pubmed":PubMedProcessor, "agnews":AGNewsProcessor, "imdb":IMDBProcessor,'sst5':Sst5Processor}
    )

    output_modes = glue_output_modes
    output_modes.update(
        {"pubmed":"classification", "agnews":"classification", "imdb":"classification", "sst5":"classification"}
    )
    return processors, output_modes

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["imdb", "agnews", "pubmed", "sst5"]:
        return {"f1":f1_score(y_true=labels, y_pred=preds, average='weighted'),
                "acc": accuracy_score(y_true=labels, y_pred=preds),
                "precision":precision_score(y_true=labels, y_pred=preds, average='weighted'),
                "SensRecall":recall_score(y_true=labels, y_pred=preds, average='weighted')}
    elif task_name in glue_processors:
        return glue_compute_metrics(task_name, preds, labels)

