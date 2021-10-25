import argparse
import torch

from transformers import (
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig
)


def set_seed(args, seed):

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)


def load_pretrainedModel(args, processors, output_modes):
    args.model_type = args.model_type.lower()
    if args.head=="AutoModelForSequenceClassification":
        processor = processors[args.task_name]()
        args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    elif args.head=="AutoModelForMaskedLM":

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None
        )

        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(args.device)
    return model, tokenizer



def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument("--base_model",
        type=str,
        required=True,
        help="The base model (for active learning experiments) name or path for loading cached data"
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--new_output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument(
        "--do_lower_case", default=True, type=bool, help="Set this flag if you are using an uncased model.",
    )


    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")


    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--continue_acq", type=int, default=0, help="Continue train with the current label pool",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=100, help="random seed for initialization")


    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")


    # Additional arguments for active learning
    parser.add_argument("--sampling", type=str, help="Acquisition function for active learning.")



    parser.add_argument("--sampledindices", default=None, type=list, help="Whether to run AL.")
    parser.add_argument("--poolsize", default=50, type=int, help="initial label pool size.")
    parser.add_argument("--ensemble", default=5, type=int, help="number of models.")
    parser.add_argument("--max_acq_size", default=5, type=int, help="number of models by each seed.")
    parser.add_argument("--ini_label_seed", default=100, type=int, help="fix the initial label pool by each seed.")
    parser.add_argument("--ensemble_model_seed", default=1234, type=int, help="fix the initial label pool by each seed.")
    parser.add_argument("--dynamic_vad", default=10, type=int, help="initial validation size by each seed.")
    parser.add_argument("--initrsize", default=20, type=int, help="initial training size")
    parser.add_argument("--acq_batch_size", default=1, type=int, help="acquisition batch size")


    args = parser.parse_args()

    # Setup CPU or GPU training

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    setattr(args, 'device', device)
    return args
