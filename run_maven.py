import argparse
import logging
import os
import random
import shutil

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from bert_crf import BertCompressedCRFForTokenClassification
from utils_maven import convert_examples_to_features, get_labels, read_examples_from_file

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "maven")

MODEL_CLASSES = {
    "bertcompressedcrf": (BertConfig, BertCompressedCRFForTokenClassification, BertTokenizer),
}


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _format_eval_prefix(mode, prefix=""):
    return f"{mode} | {prefix}" if prefix else mode


def _move_batch_to_device(batch, device):
    return tuple(t.to(device) for t in batch)


def _build_model_inputs(args, batch):
    return {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "labels": batch[3],
    }


def _extract_sentence_rows(labels_tensor, preds_tensor):
    labels_cpu = labels_tensor.detach().cpu().numpy()
    preds_cpu = preds_tensor.detach().cpu().numpy()
    return list(zip(labels_cpu, preds_cpu))


def _actual_model(model):
    return model.module if hasattr(model, "module") else model


def _assert_finite_model_parameters(model, context):
    actual = _actual_model(model)
    for name, parameter in actual.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                "Non-finite model parameter detected during {}: {}".format(context, name)
            )


def _assert_finite_tensor(tensor, context):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Non-finite tensor detected during {}.".format(context))


def _cache_file(args, mode):
    model_key = list(filter(None, args.model_name_or_path.split("/"))).pop()
    return os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(mode, model_key, str(args.max_seq_length)),
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _build_optimizer_and_scheduler(args, model, t_total, global_step=0):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler_kwargs = {
        "num_warmup_steps": args.warmup_steps,
        "num_training_steps": t_total,
    }
    if global_step > 0:
        for group in optimizer.param_groups:
            group["initial_lr"] = args.learning_rate
        scheduler_kwargs["last_epoch"] = global_step
    scheduler = get_linear_schedule_with_warmup(optimizer, **scheduler_kwargs)
    return optimizer, scheduler


def _save_best_checkpoint(args, model, tokenizer, results, epoch_idx, num_epochs, global_step, t_total):
    best_checkpoint_dir = os.path.join(args.output_dir, "best-checkpoint")
    os.makedirs(best_checkpoint_dir, exist_ok=True)
    _assert_finite_model_parameters(model, "saving best-checkpoint")
    model_to_save = _actual_model(model)
    logger.info(
        "Saving best-checkpoint | epoch=%d/%d | global_step=%d/%d | output_dir=%s",
        epoch_idx + 1,
        num_epochs,
        global_step,
        t_total,
        best_checkpoint_dir,
    )
    model_to_save.save_pretrained(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)
    torch.save(args, os.path.join(best_checkpoint_dir, "training_args.bin"))
    with open(os.path.join(best_checkpoint_dir, "best_valid_results.txt"), "w") as writer:
        writer.write("epoch = {}\n".format(epoch_idx + 1))
        writer.write("global_step = {}\n".format(global_step))
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
    )

    optimizer_steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (args.max_steps + optimizer_steps_per_epoch - 1) // optimizer_steps_per_epoch
    else:
        t_total = optimizer_steps_per_epoch * int(args.num_train_epochs)

    effective_logging_steps = args.logging_steps
    effective_save_steps = args.save_steps
    if args.evals_per_epoch > 0:
        effective_logging_steps = max(1, optimizer_steps_per_epoch // args.evals_per_epoch)
    if args.saves_per_epoch > 0:
        effective_save_steps = max(1, optimizer_steps_per_epoch // args.saves_per_epoch)

    optimizer, scheduler = _build_optimizer_and_scheduler(args, model, t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", int(args.num_train_epochs))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Optimizer steps per epoch = %d", optimizer_steps_per_epoch)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Effective logging_steps = %d", effective_logging_steps)
    logger.info("  Effective save_steps = %d", effective_save_steps)

    best_dev_f1 = float("-inf")
    no_improve_evals = 0
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    last_prune_step = 0
    should_stop_early = False
    num_epochs = int(args.num_train_epochs)

    model.zero_grad()
    set_seed(args)
    train_iterator = trange(num_epochs, desc="Epoch")
    for epoch_idx in train_iterator:
        logger.info(
            "----- Epoch %d/%d started: dataloader_batches=%d, optimizer_steps_per_epoch=%d -----",
            epoch_idx + 1,
            num_epochs,
            len(train_dataloader),
            optimizer_steps_per_epoch,
        )
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = _move_batch_to_device(batch, args.device)
            inputs = _build_model_inputs(args, batch)
            outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            _assert_finite_tensor(loss, "training loss before backward")
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            _assert_finite_tensor(torch.as_tensor(grad_norm, device=args.device), "gradient clipping")
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if effective_logging_steps > 0 and global_step % effective_logging_steps == 0:
                _assert_finite_model_parameters(model, "optimizer step {}".format(global_step))
                avg_loss = (tr_loss - logging_loss) / effective_logging_steps
                logger.info(
                    "Train progress | epoch=%d/%d | iteration=%d/%d | global_step=%d/%d | avg_loss=%.6f | lr=%.8f",
                    epoch_idx + 1,
                    num_epochs,
                    step + 1,
                    len(train_dataloader),
                    global_step,
                    t_total,
                    avg_loss,
                    scheduler.get_last_lr()[0],
                )
                if args.evaluate_during_training:
                    eval_prefix = (
                        f"trigger=logging_steps | epoch={epoch_idx + 1}/{num_epochs} | "
                        f"global_step={global_step}/{t_total} | iteration={step + 1}/{len(train_dataloader)}"
                    )
                    results, _ = evaluate(
                        args,
                        model,
                        tokenizer,
                        labels,
                        pad_token_label_id,
                        mode="valid",
                        prefix=eval_prefix,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(
                        "Validation summary | epoch=%d/%d | global_step=%d/%d | f1=%.6f | precision=%.6f | recall=%.6f | loss=%.6f",
                        epoch_idx + 1,
                        num_epochs,
                        global_step,
                        t_total,
                        results["f1"],
                        results["precision"],
                        results["recall"],
                        results["loss"],
                    )
                    improved = results["f1"] > (best_dev_f1 + args.early_stop_min_delta)
                    if improved:
                        logger.info(
                            "New best validation F1 improved from %.6f to %.6f",
                            best_dev_f1,
                            results["f1"],
                        )
                        best_dev_f1 = results["f1"]
                        no_improve_evals = 0
                        _save_best_checkpoint(
                            args, model, tokenizer, results, epoch_idx, num_epochs, global_step, t_total
                        )
                    elif args.early_stop_patience > 0:
                        no_improve_evals += 1
                        logger.info(
                            "No dev F1 improvement beyond min_delta=%.6f | no_improve_evals=%d/%d",
                            args.early_stop_min_delta,
                            no_improve_evals,
                            args.early_stop_patience,
                        )
                        if no_improve_evals >= args.early_stop_patience:
                            logger.info(
                                "Early stopping triggered at epoch=%d/%d, global_step=%d/%d",
                                epoch_idx + 1,
                                num_epochs,
                                global_step,
                                t_total,
                            )
                            should_stop_early = True
                logging_loss = tr_loss

            if effective_save_steps > 0 and global_step % effective_save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                os.makedirs(output_dir, exist_ok=True)
                _assert_finite_model_parameters(model, "saving checkpoint")
                model_to_save = _actual_model(model)
                logger.info("Saving model checkpoint to %s", output_dir)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if args.prune_ffn and args.prune_interval > 0 and global_step - last_prune_step >= args.prune_interval:
                logger.info("Performing FFN pruning at step %d (prune_ratio=%.4f)", global_step, args.prune_ratio)
                actual_model = _actual_model(model)
                actual_model._register_ffn_hooks()
                with torch.enable_grad():
                    prune_inputs = _build_model_inputs(args, batch)
                    prune_outputs = actual_model(pad_token_label_id=pad_token_label_id, **prune_inputs)
                    prune_loss = prune_outputs[0]
                    _assert_finite_tensor(prune_loss, "FFN pruning loss")
                    prune_loss.backward()
                actual_model.prune_ffn(args.prune_ratio)
                model.zero_grad()
                optimizer, scheduler = _build_optimizer_and_scheduler(args, actual_model, t_total, global_step)
                logger.info("FFN pruning completed. Optimizer and scheduler rebuilt.")
                last_prune_step = global_step

            if should_stop_early:
                epoch_iterator.close()
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break

        logger.info("----- Epoch %d/%d finished: completed_global_step=%d/%d -----",
                    epoch_idx + 1, num_epochs, global_step, t_total)
        if should_stop_early or (args.max_steps > 0 and global_step >= args.max_steps):
            train_iterator.close()
            break

    return global_step, tr_loss / max(1, global_step)


def _evaluate_once(args, model, eval_dataset, labels, pad_token_label_id, eval_label):
    _assert_finite_model_parameters(model, "evaluation {}".format(eval_label))
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=args.eval_batch_size,
    )

    logger.info("***** Running evaluation: %s *****", eval_label)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    sentence_rows = []
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = _move_batch_to_device(batch, args.device)
        with torch.no_grad():
            inputs = _build_model_inputs(args, batch)
            outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            tmp_eval_loss = outputs[0]
            best_path = outputs[-1]
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            _assert_finite_tensor(tmp_eval_loss, "evaluation loss")
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        sentence_rows.extend(_extract_sentence_rows(inputs["labels"], best_path))

    eval_loss = eval_loss / max(1, nb_eval_steps)
    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(len(sentence_rows))]
    preds_list = [[] for _ in range(len(sentence_rows))]
    for row_idx, (label_row, pred_row) in enumerate(sentence_rows):
        for label_id, pred_id in zip(label_row, pred_row):
            if int(label_id) != pad_token_label_id:
                out_label_list[row_idx].append(label_map[int(label_id)])
                preds_list[row_idx].append(label_map[int(pred_id)])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    logger.info("***** Eval results: %s *****", eval_label)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    return results, preds_list


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    return _evaluate_once(args, model, eval_dataset, labels, pad_token_label_id, _format_eval_prefix(mode, prefix))


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    cached_features_file = _cache_file(args, mode)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=pad_token_label_id,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def main():
    parser = argparse.ArgumentParser(
        description="TEBF standalone training and evaluation entry point."
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)

    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--save_steps", default=50, type=int)
    parser.add_argument("--evals_per_epoch", default=0, type=int)
    parser.add_argument("--saves_per_epoch", default=0, type=int)
    parser.add_argument("--early_stop_patience", default=0, type=int)
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--use_attention_pooling", action="store_true")
    parser.add_argument("--attention_pooling_variant", default="multihead", choices=["multihead"])
    parser.add_argument("--attention_pooling_heads", default=4, type=int)
    parser.add_argument("--use_sentence_event_fusion", action="store_true")

    parser.add_argument("--compression_method", default=None, choices=["gate"])
    parser.add_argument("--tsvd_dim", default=256, type=int)
    parser.add_argument("--gate_init_bias", default=2.197, type=float)
    parser.add_argument("--gate_sparsity_lambda", default=1e-4, type=float)

    parser.add_argument("--prune_ffn", action="store_true")
    parser.add_argument("--prune_ratio", default=0.3, type=float)
    parser.add_argument("--prune_interval", default=500, type=int)

    args = parser.parse_args()
    args.model_type = args.model_type.lower()
    args.data_dir = os.path.abspath(args.data_dir)
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.max_steps = -1
    args.overwrite_cache = False
    args.evaluate_during_training = args.evals_per_epoch > 0

    if args.model_type != "bertcompressedcrf":
        raise ValueError("TEBF only supports --model_type bertcompressedcrf.")
    if not os.path.isdir(args.data_dir):
        raise ValueError("MAVEN data directory does not exist: {}".format(args.data_dir))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1.")
    if args.use_sentence_event_fusion and not args.use_attention_pooling:
        raise ValueError("--use_sentence_event_fusion requires --use_attention_pooling.")
    if args.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be >= 0.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be >= 0.")
    if not 0.0 <= args.prune_ratio < 1.0:
        raise ValueError("--prune_ratio must be in [0, 1).")

    cleared_output_dir = None
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        if not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir.".format(
                    args.output_dir
                )
            )
        protected_dirs = {
            os.path.abspath(os.sep),
            os.path.abspath(os.path.expanduser("~")),
            os.path.abspath(PROJECT_ROOT),
            os.path.abspath(args.data_dir),
        }
        output_dir = os.path.abspath(args.output_dir)
        if output_dir in protected_dirs:
            raise ValueError("Refusing to clear protected output directory: {}".format(output_dir))
        shutil.rmtree(output_dir)
        cleared_output_dir = output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[TqdmLoggingHandler()],
        force=True,
    )
    logging.captureWarnings(True)
    if cleared_output_dir is not None:
        logger.info("Cleared existing output directory because --overwrite_output_dir is set: %s", cleared_output_dir)
    if args.evals_per_epoch > 0:
        logger.info("Auto-enabled evaluate_during_training because evals_per_epoch=%d", args.evals_per_epoch)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, 16-bits training: %s",
        -1,
        device,
        args.n_gpu,
        False,
    )

    set_seed(args)
    labels = get_labels("")
    num_labels = len(labels)
    pad_token_label_id = -100

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    if args.compression_method is not None and not 0 < args.tsvd_dim <= min(config.vocab_size, config.hidden_size):
        raise ValueError(
            "--tsvd_dim must be in [1, {}], got {}.".format(
                min(config.vocab_size, config.hidden_size),
                args.tsvd_dim,
            )
        )
    config.use_attention_pooling = args.use_attention_pooling
    config.attention_pooling_variant = args.attention_pooling_variant
    config.attention_pooling_heads = args.attention_pooling_heads
    config.use_sentence_event_fusion = args.use_sentence_event_fusion
    config.sentence_event_fusion_gate_bias = 3.0
    config.num_maven_types = (num_labels - 1) // 2

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to(args.device)

    if args.compression_method is not None:
        actual_model = model.module if hasattr(model, "module") else model
        compression_already_loaded = (
            getattr(actual_model, "compressed_embedding", None) is not None
            and getattr(actual_model, "is_decomposed", False)
            and getattr(actual_model, "compression_method", None) == args.compression_method
            and getattr(actual_model, "tsvd_dim", None) == args.tsvd_dim
        )
        if compression_already_loaded:
            logger.info(
                "Compression already restored from checkpoint: method=%s, tsvd_dim=%d",
                args.compression_method,
                args.tsvd_dim,
            )
        else:
            actual_model.init_compression(
                method=args.compression_method,
                tsvd_dim=args.tsvd_dim,
                gate_init_bias=args.gate_init_bias,
                gate_sparsity_lambda=args.gate_sparsity_lambda,
            )
            actual_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        _assert_finite_model_parameters(model, "saving final checkpoint")
        model_to_save = _actual_model(model)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}
    if args.do_eval:
        os.makedirs(args.output_dir, exist_ok=True)
        best_checkpoint_dir = os.path.join(args.output_dir, "best-checkpoint")
        if args.do_train:
            eval_target = best_checkpoint_dir if os.path.exists(best_checkpoint_dir) else args.output_dir
        else:
            eval_target = args.model_name_or_path
        logger.info("Evaluate the best available checkpoint on dev: %s", eval_target)
        tokenizer = tokenizer_class.from_pretrained(eval_target, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(eval_target)
        model.to(args.device)
        result, _ = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="dev",
            prefix="final-dev-eval",
        )
        results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    return results


if __name__ == "__main__":
    main()
