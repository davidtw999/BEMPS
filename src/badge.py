import numpy as np
import torch
from torch.utils.data import Subset, SequentialSampler, DataLoader
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, Softmax
from torch.nn.functional import one_hot
import pathlib
import os
from sklearn.cluster import KMeans

def badge_gradient(model, inputs, **kwargs):
    """Return the loss gradient with respect to the penultimate layer for BADGE"""
    pooled_output = bert_embedding(model, inputs)
    logits = model.classifier(pooled_output)
    batch_size, num_classes = logits.size()

    softmax = Softmax(dim=1)
    probs = softmax(logits)
    preds = probs.argmax(dim=1)
    preds_oh = one_hot(preds, num_classes=num_classes)
    scales = probs - preds_oh

    grads_3d = torch.einsum('bi,bj->bij', scales, pooled_output)
    grads = grads_3d.view(batch_size, -1)
    return grads


def bert_embedding(model, inputs, **kwargs):
    """Return the [CLS] embedding for each input in [inputs]"""

    inputs.pop("masked_lm_labels", None)
    bert_output = model.distilbert(**inputs)[0]
    bert_output = torch.mean(bert_output, dim=1)

    return bert_output



def batch_scores_or_vectors(batch, args, model, tokenizer):

    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {}


    inputs["input_ids"] = batch[0]
    inputs["attention_mask"] = batch[1]
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
        )

    with torch.no_grad():
        scores_or_vectors = badge_gradient(model, inputs)
        # scores_or_vectors = sampling_method(args.sampling)(model=model, inputs=inputs)


    return scores_or_vectors



def get_scores_or_vectors_badge(eval_dataset, args, model, tokenizer=None):

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)

    for eval_task in eval_task_names:

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

        all_scores_or_vectors = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            scores_or_vectors = batch_scores_or_vectors(batch, args, model, tokenizer)

            if all_scores_or_vectors is None:
                all_scores_or_vectors = scores_or_vectors.detach().cpu().numpy()
            else:
                all_scores_or_vectors = np.append(all_scores_or_vectors, scores_or_vectors.detach().cpu().numpy(), axis=0)


    all_scores_or_vectors = torch.tensor(all_scores_or_vectors)

    return all_scores_or_vectors