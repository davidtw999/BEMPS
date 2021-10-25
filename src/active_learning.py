6# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import logging
import os
import math
import numpy as np
import torch
import sys
import random
import copy
import src.setup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from src.data import (
    convert_examples_to_features,
    compute_metrics,
    get_processor,
)

from src.queryStrategy import (
    bald,
    mocu_wmocu,
    bemps_coremse,
    bemps_corelog,
    max_entropy_acquisition_function,
    random_queries,
    bemps_coremse_batch,
    mocu_wmocu_batch,
    bemps_corelog_batch,
    random_queries_batch,
    lm_clustering,
    bemps_coremse_batch_topk,
    bemps_corelog_batch_topk,
    badge_clustering
)

from src.badge import (
    get_scores_or_vectors_badge
)


logger = logging.getLogger(__name__)

def train_model(args, train_dataset, model, tokenizer, acqIdx, va_dataset, processors, output_modes):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )


    best_loss = sys.maxsize
    early_break_count = 0

    for cur_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):


            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )
            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        results, _ = evaluate_model(args, model, tokenizer, 'evaluate', va_dataset,
                              processors=processors, output_modes=output_modes)

        cur_evaLoss = results['loss']

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if cur_evaLoss <= best_loss:

            cur_ep = str(cur_epoch + 1)

            model_best = copy.deepcopy(model)

            best_results = {"Acquired sample":acqIdx,"validation set": results,"ave tr loss":tr_loss / global_step,"epoches":cur_ep}
            best_loss = cur_evaLoss

        else:
            early_break_count = early_break_count + 1

        if early_break_count > 5:
            break


    results_test, logits_test = evaluate_model(args, model_best, tokenizer,'test', va_dataset,
                                         processors=processors, output_modes=output_modes)
    torch.save(logits_test, os.path.join(args.new_output_dir, 'testTensor.pt'))


    if args.sampling != 'badge':
        _, logits_unlabeled = evaluate_model(args, model_best, tokenizer, 'unlabeled', va_dataset,
                                       processors=processors, output_modes=output_modes)
        torch.save(logits_unlabeled, os.path.join(args.new_output_dir, 'unlabeledTensor.pt'))


    write_result_to_txt(args.new_output_dir, best_results)
    write_result_to_txt(args.new_output_dir, {"Acquired sample":acqIdx,"test set": results_test})


    return global_step, tr_loss / global_step




def write_result_to_txt(save_path, results):
    fileName = save_path + "/result.txt"
    file = open(fileName, "a+")
    results = str(results) + "\n"
    file.writelines([results])
    file.close()


def save_eval_txt(output_dir, results):

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(''))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))



def evaluate_model(args, model, tokenizer, dataType, va_dataset, prefix="", processors=None, output_modes=None):

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if dataType == 'evaluate':
            eval_dataset = va_dataset
            prefix = prefix + 'validataion set'

        elif dataType == 'test':
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, test=True,
                                                   processors=processors, output_modes=output_modes)
            prefix = prefix + 'test set'
        elif dataType == 'unlabeled':
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False, test=False,
                                                   processors=processors, output_modes=output_modes)
            unlabeledSample_file = os.path.join(args.output_dir, "unLabelPool.pt")
            unlabeledSample = torch.load(unlabeledSample_file)
            eval_dataset = Subset(eval_dataset, unlabeledSample)
            prefix = prefix + 'unlabeled pool'

        if not os.path.exists(eval_output_dir) :
            os.makedirs(eval_output_dir)


        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)


        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logger.info(f"eval_loss = {eval_loss}")

        if args.output_mode == "classification":
            logits = preds
            preds = np.argmax(preds, axis=1)

        np.seterr(divide='ignore', invalid='ignore')
        results = compute_metrics(eval_task, preds, out_label_ids)
        np.seterr(divide='warn', invalid='warn')
        results.update({"loss":eval_loss})


    return results, logits


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False, processors=None, output_modes=None):

    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    if test:
        data_split = "test"
    elif evaluate:
        data_split = "dev"
    else:
        data_split = "train"

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            data_split,
            list(filter(None, args.base_model.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )


    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:

            label_list[1], label_list[2] = label_list[2], label_list[1]
        if test:
            examples = processor.get_test_examples(args.data_dir)
        elif evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)


    # print(all_labels)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset



def generate_random_indices(train_dataset, seed, initial_labelPoolSize):

    max_len = len(train_dataset)
    indices = range(max_len)
    random.seed(seed)
    label_index = random.sample(indices, initial_labelPoolSize)
    unlabeled_index = list(set(indices) - set(label_index))
    label_index = torch.LongTensor(label_index)
    unlabeled_index = torch.LongTensor(unlabeled_index)
    return label_index, unlabeled_index



def generate_balance_val_indices(val_dataset, seed, val_size):
    max_len = len(val_dataset)
    indices = range(max_len)
    labels = val_dataset[:][-1]
    _, label_index, _, _ = train_test_split(indices, labels, test_size=val_size, stratify=labels, random_state=seed)

    return Subset(val_dataset, label_index)



def generate_balance_random_indices(train_dataset, seed, tol_size, tr_size):


    max_len = len(train_dataset)
    indices = range(max_len)
    labels = train_dataset[:][-1]

    rest_indices, tr_indices, rest_label, _ = train_test_split(indices, labels, test_size=tr_size, stratify=labels, random_state=seed)
    va_size = tol_size - tr_size

    _, va_indices, _, _ = train_test_split(rest_indices, rest_label, test_size=va_size, stratify=rest_label, random_state=seed)

    label_index = tr_indices + va_indices
    unlabeled_index = list(set(indices) - set(label_index))

    label_index = torch.LongTensor(label_index)
    unlabeled_index = torch.LongTensor(unlabeled_index)
    return label_index, unlabeled_index


def split_train_val_indices(label_indices, train_dataset):

    shuffeled_idx = label_indices[torch.randperm(label_indices.size()[0])]

    slice_split = math.ceil(len(shuffeled_idx) * 0.75)

    tr_dataset = Subset(train_dataset, shuffeled_idx[:slice_split])
    va_dataset = Subset(train_dataset, shuffeled_idx[slice_split:])

    return tr_dataset, va_dataset


def main():
    args = src.setup.get_arguments()

    if (os.path.isfile(os.path.join(args.output_dir, 'result.txt'))
            and args.continue_acq == 0):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --continue_acq to continue training.".format(
                args.output_dir
            )
        )

    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            probPath = os.path.join(args.output_dir, 'probsEnsemble')
            lpPath = os.path.join(args.output_dir, 'labelPoolckpt')
            os.makedirs(probPath)
            os.makedirs(lpPath)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process device: %s, n_gpu: %s",
        args.device,
        args.n_gpu,
    )

    # Set seed
    src.setup.set_seed(args, args.seed)

    processors, output_modes = get_processor()

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    if args.sampling != "alps":
        args.head = "AutoModelForSequenceClassification"
    else:
        args.head = 'AutoModelForMaskedLM'
    model, tokenizer = src.setup.load_pretrainedModel(args, processors, output_modes)

    # model1 = copy.deepcopy(model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False,
                                            processors=processors, output_modes=output_modes)

    if args.continue_acq == 0:

        ## load tensordataset object by passing tokenizer ebeddmming


        label_indices, unlabeled_indices = generate_balance_random_indices(train_dataset, args.ini_label_seed,
                                                                           args.poolsize, args.initrsize)
        torch.save(label_indices, os.path.join(args.output_dir, 'labeledPool.pt'))
        torch.save(unlabeled_indices, os.path.join(args.output_dir, 'unLabelPool.pt'))


        write_result_to_txt(args.output_dir, {"max_seq_length": args.max_seq_length,
                                              "num_train_epochs": args.num_train_epochs,
                                              "Initial label pool": list(label_indices.numpy()),
                                              "Initial unlabeled pool": list(unlabeled_indices.numpy())})


        for eachAcquire in range(0, args.max_acq_size,args.acq_batch_size) :

            acqIdx = str(eachAcquire + args.acq_batch_size)

            sampled_file = os.path.join(args.output_dir, 'labeledPool.pt')

            if os.path.isfile(sampled_file):
                label_indices = torch.load(sampled_file)
            else:
                raise ValueError("Sampled_dataset not found: %s" % ('labeledPool.pt'))

            temp_seed = args.seed
            scores = []

            for eachM in range(args.ensemble):

                model1 = copy.deepcopy(model)
                tr_dataset, va_dataset = split_train_val_indices(label_indices, train_dataset)

                logger.info("\n\nTRAINING WITH THE MODEL %s \n\n", str(eachM + 1))
                args.new_output_dir = os.path.join(args.output_dir, str(eachM + 1) + '_' + str(temp_seed))
                if not os.path.exists(args.new_output_dir):
                    os.makedirs(args.new_output_dir)

                global_step, tr_loss = train_model(args, tr_dataset, model1, tokenizer, acqIdx, va_dataset,
                                             processors, output_modes)
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

                temp_seed = temp_seed + args.ensemble_model_seed

                if (args.sampling == 'badge'):
                    scores_or_vectors = get_scores_or_vectors_badge(train_dataset, args, model1, tokenizer)
                    scores.append(scores_or_vectors)


            logger.info("\n\nACQUIRING %sTH SAMPLE \n\n", acqIdx)

            probsTest_X_E_Y = get_probs_from_ensembleModels(args, 'testTensor.pt', args.ensemble_model_seed)
            torch.save(probsTest_X_E_Y, os.path.join(probPath, 'testProbsTensor_' + str(acqIdx) + '.pt'))

            ## add wray's metric here
            res = compute_mean_metrics(args, probsTest_X_E_Y, processors, output_modes)

            labeled_indices = get_indices(args, 'labeledPool.pt')
            unlabel_indices = get_indices(args, 'unLabelPool.pt')

            if (args.sampling == 'badge'):
                probsULP_X_E_Y = torch.mean(torch.stack(scores), dim=0)
                probsULP_X_E_Y = probsULP_X_E_Y[unlabel_indices]
            else:
                probsULP_X_E_Y = get_probs_from_ensembleModels(args, 'unlabeledTensor.pt', args.ensemble_model_seed)
                torch.save(probsULP_X_E_Y, os.path.join(probPath, 'unLBProbsTensor_' + str(acqIdx) + '.pt'))


            sampled_index = query_method(probsULP_X_E_Y, args.sampling, args.acq_batch_size)

            updated_lab_idxs, updated_unlab_idxs = update_labeled_and_unlabeled_pool(sampled_index, labeled_indices.tolist(),
                                                                                     unlabel_indices.tolist())
            write_result_to_txt(args.output_dir, {"Acquired sample":acqIdx, "Ensemble test set": res, "sampled_index": updated_lab_idxs[-args.acq_batch_size:],
                                                  "len tr":len(tr_dataset),"len va":len(va_dataset)})
            torch.save(torch.LongTensor(updated_lab_idxs), os.path.join(args.output_dir, 'labeledPool.pt'))
            torch.save(torch.LongTensor(updated_unlab_idxs), os.path.join(args.output_dir, 'unLabelPool.pt'))

            torch.save(torch.LongTensor(updated_lab_idxs), os.path.join(lpPath, 'labeledPool_' + str(acqIdx) + '.pt'))
            torch.save(torch.LongTensor(updated_unlab_idxs), os.path.join(lpPath, 'unLabelPool_' + str(acqIdx) + '.pt'))


    ## two optimal settings: inner + kmean, outer +kmean_pp
    elif args.continue_acq == 1:

        probPath = os.path.join(args.output_dir, 'probsEnsemble')
        lpPath = os.path.join(args.output_dir, 'labelPoolckpt')

        if not os.path.exists(probPath):
            os.makedirs(probPath)
        if not os.path.exists(lpPath):
            os.makedirs(lpPath)


        for eachAcquire in range(0, args.max_acq_size, args.acq_batch_size):

            sampled_file = os.path.join(args.output_dir, 'labeledPool.pt')

            if os.path.isfile(sampled_file):
                label_indices = torch.load(sampled_file)

                acqIdx = str(len(label_indices) - args.poolsize + args.acq_batch_size)

                # sampled_dataset = Subset(train_dataset, label_indices)
                # tr_dataset, va_dataset = split_train_val_indices(label_indices, train_dataset, args, acqIdx)
            else:
                raise ValueError("Sampled_dataset not found: %s" % ('labeledPool.pt'))



            temp_seed = args.seed

            for eachM in range(args.ensemble):

                tr_dataset, va_dataset = split_train_val_indices(label_indices, train_dataset)

                logger.info("\n\nTRAINING WITH THE MODEL %s \n\n", str(eachM + 1))
                args.new_output_dir = os.path.join(args.output_dir, str(eachM + 1) + '_' + str(temp_seed))
                if not os.path.exists(args.new_output_dir):
                    os.makedirs(args.new_output_dir)


                global_step, tr_loss = train_model(args, tr_dataset, model1, tokenizer, acqIdx, va_dataset)
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

                temp_seed = temp_seed + args.ensemble_model_seed
                src.setup.set_seed(args, temp_seed)  # Added here for reproductibility

                model1 = copy.deepcopy(model)


            logger.info("\n\nACQUIRING %sTH SAMPLE \n\n", acqIdx)


            probsTest_X_E_Y = get_probs_from_ensembleModels(args, 'testTensor.pt', args.ensemble_model_seed)

            torch.save(probsTest_X_E_Y, os.path.join(probPath, 'testProbsTensor_' + str(acqIdx) + '.pt'))


            ## add wray's metric here
            res = compute_mean_metrics(args, probsTest_X_E_Y)

            probsULP_X_E_Y = get_probs_from_ensembleModels(args, 'unlabeledTensor.pt', args.ensemble_model_seed)
            # print(probsULP_X_E_Y.shape)
            torch.save(probsULP_X_E_Y, os.path.join(probPath, 'unLBProbsTensor_' + str(acqIdx) + '.pt'))

            sampled_index = query_method(probsULP_X_E_Y, args.sampling, args.acq_batch_size)

            labeled_indices = get_indices(args, 'labeledPool.pt')
            unlabel_indices = get_indices(args, 'unLabelPool.pt')
            # print(sampled_index)

            updated_lab_idxs, updated_unlab_idxs = update_labeled_and_unlabeled_pool(sampled_index, labeled_indices.tolist(),
                                                                                     unlabel_indices.tolist())
            write_result_to_txt(args.output_dir, {"Acquired sample": acqIdx, "Ensemble test set": res,
                                                  "sampled_index": updated_lab_idxs[-args.acq_batch_size:],
                                                  "len tr": len(tr_dataset), "len va": len(va_dataset)})
            torch.save(torch.LongTensor(updated_lab_idxs), os.path.join(args.output_dir, 'labeledPool.pt'))
            torch.save(torch.LongTensor(updated_unlab_idxs), os.path.join(args.output_dir, 'unLabelPool.pt'))

            torch.save(torch.LongTensor(updated_lab_idxs), os.path.join(lpPath, 'labeledPool_' + str(acqIdx) + '.pt'))
            torch.save(torch.LongTensor(updated_unlab_idxs), os.path.join(lpPath, 'unLabelPool_' + str(acqIdx) + '.pt'))


def get_indices(args, fileName):

    fileinDir = os.path.join(args.output_dir, fileName)

    if os.path.isfile(fileinDir):
        indices = torch.load(fileinDir)
    else:
        raise ValueError("Task not found: %s" % (fileinDir))
    return indices


def update_labeled_and_unlabeled_pool(sampled_index, label_index, unlabeled_index):
    sampled_unlabeled_element = [unlabeled_index[x] for x in sampled_index]
    # print(sampled_unlabeled_element)
    label_index_new = label_index + sampled_unlabeled_element

    for x in sampled_unlabeled_element:
        unlabeled_index.remove(x)
    return label_index_new, unlabeled_index





def query_method(prob_X_E_Y, sampling, batch_size):

    if batch_size == 1:
        if sampling == 'coremse':
            rr = bemps_coremse(prob_X_E_Y, 0.3)
        elif sampling == 'corelog':
            rr = bemps_corelog(prob_X_E_Y, 0.3)
        elif sampling == 'uncertainty':
            rr = max_entropy_acquisition_function(prob_X_E_Y)
        elif sampling == 'rand':
            rr = random_queries(prob_X_E_Y.shape[0])

        if sampling != 'rand':
            winner_q_scores, winner_index = rr.max(0)
            winner_index = np.array([winner_index.item()])
        else:
            winner_index = np.array([rr])
        return winner_index

    else:
        if sampling == 'coremsebatch':
            winner_index = bemps_coremse_batch(prob_X_E_Y, batch_size, 0.3, 0.5)
            return winner_index
        elif sampling == 'corelogbatch':
            winner_index = bemps_corelog_batch(prob_X_E_Y, batch_size, 0.3, 0.5)
            return winner_index
        elif sampling == 'randbatch':
            winner_index = random_queries_batch(prob_X_E_Y.shape[0], batch_size)
            return winner_index
        elif sampling == 'unctybatch':
            rr = max_entropy_acquisition_function(prob_X_E_Y)
            winner_index = rr.topk(batch_size).indices.numpy()
            return winner_index
        elif sampling == 'coremsetopk':
            winner_index = bemps_coremse_batch_topk(prob_X_E_Y, batch_size, 0.3)
            return winner_index
        elif sampling == 'corelogtopk':
            winner_index = bemps_corelog_batch_topk(prob_X_E_Y, batch_size, 0.3)
            return winner_index

        else:
            print("Wrong bug")
            return False




def compute_mean_metrics(args, probs_X_E_Y, processors, output_modes):
    ## use the mean prob to classify, or we can use majority vote later
    probsMean_X_Y = torch.mean(probs_X_E_Y, dim=1)
    preds = torch.argmax(probsMean_X_Y, dim=-1)

    if args.task_name in ["imdb", "agnews", "pubmed", "sst5"]:
        test_dataset = load_and_cache_examples(args, args.task_name, '', evaluate=False, test=True,
                                               processors=processors, output_modes=output_modes)
        labels = test_dataset[:][-1]

    b_score = []
    p_score = []

    for i in range(probsMean_X_Y.shape[0]):
        true_labels = labels.tolist()
        p_score.append(probsMean_X_Y[i,true_labels[i]].item())
        b_score.append(1 - probsMean_X_Y[i,true_labels[i]].item())

    b_score = np.mean(np.power(np.array(b_score), 2))
    p_score = np.mean(np.log(np.array(p_score)))


    return {"f1": f1_score(y_true=labels, y_pred=preds, average='weighted'),
                "acc": accuracy_score(y_true=labels, y_pred=preds), "b_score":b_score, "p_score":p_score,
                "precision": precision_score(y_true=labels, y_pred=preds, average='weighted'),
                "SensRecall": recall_score(y_true=labels, y_pred=preds, average='weighted')}



def get_probs_from_ensembleModels(args, fileName, addSeed):

    logits = []
    tempSeed=args.seed

    for i in range(args.ensemble):
        subRoot = os.path.join(args.output_dir, str(i + 1)+'_'+str(tempSeed))
        fileinDir = os.path.join(subRoot, fileName)

        if os.path.isfile(fileinDir):
            logits.append(torch.load(fileinDir))

        tempSeed = tempSeed + addSeed

    logits_E_X_Y = np.stack(logits, axis=0)
    probs_E_X_Y = torch.softmax(torch.tensor(logits_E_X_Y), dim=-1)
    probs_X_E_Y = probs_E_X_Y.transpose(0, 1)
    return probs_X_E_Y


if __name__ == "__main__":
    main()
