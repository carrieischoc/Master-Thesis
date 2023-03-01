import os
import argparse
import random
import torch
from datasets import Dataset
from pytorch_pretrained_bert import BertConfig
from preprocess.utils import base_path
from preprocess.filter import select_ds_column
from BertSum.src.prepro.data_builder import BertData
from BertSum.src.models import model_builder, data_loader
from BertSum.src.models.trainer import build_trainer


def generate_bertdata(args, base_path: str):
    bert = BertData(args)
    path = os.path.join(
        base_path,
        args.dataset[0],
        args.split[0],
        f"baselines/intermediate_{args.extend_method}",
    )
    dataset = Dataset.load_from_disk(path)
    kwargs_dict = {"bert": bert}
    dataset = dataset.map(format_to_bert_map, fn_kwargs=kwargs_dict, num_proc=16)
    dataset_dict = []
    for ex in dataset:
        dataset_dict.append(ex["b_data_dict"])

    path = os.path.join(
        base_path, args.dataset[0], args.split[0], "extraction/bertdata/"
    )

    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
    torch.save(dataset_dict, f"{path}{args.extend_method}.pt")


def format_to_bert_map(example, bert):

    b_data = bert.preprocess(
        example["source"], example["target"], example["intermediate_summary_indices"]
    )
    if b_data is None:
        indexed_tokens = labels = segments_ids = cls_ids = src_txt = tgt_txt = None
    else:
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    b_data_dict = {
        "src": indexed_tokens,
        "labels": labels,
        "segs": segments_ids,
        "clss": cls_ids,
        "src_txt": src_txt,
        "tgt_txt": tgt_txt,
    }
    example["b_data_dict"] = b_data_dict

    return example


def train_bertsum_model(args, device_id: int = 0):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = model_builder.Summarizer(args, "cuda", load_pretrained_bert=True)
    optim = model_builder.build_optim(args, model, None)
    trainer = build_trainer(args, device_id, model, optim)
    path = os.path.join(
        base_path,
        args.dataset[0],
        args.split[0],
        f"extraction/bertdata/{args.extend_method}.pt",
    )

    def load_bert(path):
        def _lazy_load(path):
            return torch.load(path)

        yield _lazy_load(path)

    def train_iter_fun():
        return data_loader.Dataloader(
            args, load_bert(path), args.batch_size, "cuda", shuffle=False, is_test=False
        )

    trainer.train(
        train_iter_fun,
        args.train_steps,
    )


if __name__ == "__main__":

    path = "/home/jli/working_dir/2022-jiahui-li-thesis/"
    parser = argparse.ArgumentParser(description="Get training parameters of datasets")
    parser.add_argument("-d", "--dataset", nargs=1, type=str, help="name of dataset")
    parser.add_argument("--split", nargs=1, type=str, help="split of dataset")
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-min_nsents", default=3, type=int)
    parser.add_argument("-max_nsents", default=100, type=int)
    parser.add_argument("-min_src_ntokens", default=5, type=int)
    parser.add_argument("-max_src_ntokens", default=200, type=int)
    parser.add_argument(
        "-encoder",
        default="transformer",
        type=str,
        choices=["classifier", "transformer", "rnn", "baseline"],
    )
    parser.add_argument(
        "-bert_config_path", default=f"{path}BertSum/bert_config_uncased_base.json"
    )
    parser.add_argument("-batch_size", default=1000, type=int)
    parser.add_argument("-visible_gpus", default="0", type=str)
    parser.add_argument("-gpu_ranks", default="0", type=str)

    parser.add_argument("-use_interval", default=True, type=bool)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=4, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=bool, default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default="adam", type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default="", type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=2, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-seed", default=666, type=int)
    # parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    # parser.add_argument("-bert_data_path", default='../bert_data/cnndm')
    parser.add_argument("-model_path", default="../models/")
    parser.add_argument("-result_path", default="../results")
    parser.add_argument("-temp_dir", default="../temp")
    parser.add_argument("-train_from", default="")
    parser.add_argument(
        "-extend_method",
        type=str,
        choices=["top1", "top_extended_oracle_score", "top_extended_oracle_random"],
    )

    args = parser.parse_args()
    # method in choices of 'top1', 'top_extended_oracle_score', 'top_extended_oracle_random'
    # generate_bertdata(args, base_path)
    train_bertsum_model(args)
