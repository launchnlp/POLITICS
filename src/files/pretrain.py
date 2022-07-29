import argparse
import math
import os
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from src.files.dataset import MLMDataset, MLMCollator, TupleDataset, TupleCollator
from src.files.model import POLITICS
from src.files.utils import set_seed

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel("INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--mlm_train_files", type=str, nargs='+', help="List of json files containing the MLM training data."
    )
    parser.add_argument(
        "--mlm_val_files", type=str, nargs='+', help="List of json files containing the MLM validation data."
    )
    parser.add_argument(
        "--contra_train_file", type=str, help="Json file containing the contrastive training data."
    )
    parser.add_argument(
        "--contra_val_file", type=str, help="Json file containing the contrastive validation data."
    )
    parser.add_argument(
        "--per_gpu_mlm_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the MLM training dataloader.",
    )
    parser.add_argument(
        "--per_gpu_mlm_eval_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the MLM evaluation dataloader.",
    )
    parser.add_argument(
        "--per_gpu_contra_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the contra training dataloader.",
    )
    parser.add_argument(
        "--per_gpu_contra_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the contra evaluation dataloader.",
    )
    parser.add_argument(
        "--mlm_learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use for MLM.",
    )
    parser.add_argument(
        "--contra_learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use for contra.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
        help="Total number of training epochs to perform.")
    parser.add_argument('--max_train_steps', type=int, default=2500,
        help="Maximum number of training steps.")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of gpus to use.")
    parser.add_argument("--logging_steps", type=int, default=16, help="Number of steps to log.")
    parser.add_argument("--model_name", type=str, default='roberta-base', help="Model name.")
    parser.add_argument(
        "--mlm_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass for MLM.",
    )
    parser.add_argument(
        "--contra_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass for contra.",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Where to store the final model.")
    parser.add_argument('--do_train', action='store_true', help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true', help="Whether to perform evaluation")
    parser.add_argument('--train_mlm', action='store_true',
        help="Whether to use MLM objective.")
    parser.add_argument('--train_contrast', action='store_true',
        help="Whether to use contrastive loss.")
    parser.add_argument('--contrast_loss', type=str, choices=['triplet', 'multi-negative'],
        default='triplet', help='Type of contrastive loss.') # triplet loss or infoNCE loss
    parser.add_argument("--contrast_alpha", type=float, default=0.5,
        help="alpha * ideology_loss + (1-alpha) * story_loss")
    parser.add_argument("--loss_alpha", type=float, default=0.5,
        help="alpha * MLM_loss + (1-alpha) * contrast_loss")
    parser.add_argument("--contra_num_article_limit", type=int, default=32,
        help="Limit for number of negatives in contrastive loss.")
    parser.add_argument('--use_ideo_loss', action='store_true',
        help="Whether to use ideology objective in contrastive learning.")
    parser.add_argument('--use_story_loss', action='store_true',
        help="Whether to use story objective in contrastive learning.")
    parser.add_argument("--ideo_margin", type=float, default=1.0,
        help="Margin for ideology triplet loss.")
    parser.add_argument("--story_margin", type=float, default=1.0,
        help="Margin for story triplet loss.")
    parser.add_argument("--ideo_tau", type=float, default=0.07,
        help="Temperature for ideology multiple-negative loss.")
    parser.add_argument("--story_tau", type=float, default=0.3,
        help="Temperature for story multiple-negative loss.")
    parser.add_argument('--mask_entity', action='store_true',
        help="Whether to additionally mask named entities")
    parser.add_argument('--mask_sentiment', action='store_true',
        help="Whether to additionally mask sentiment lexicon")
    parser.add_argument('--predict_ideology', action='store_true',
        help="Whether to directly predict ideology in pretraining")
    parser.add_argument("--lexicon_dir", type=str, help="Directory for sentiment lexicon.")
    parser.add_argument('--data_process_worker', type=int, default=0,
        help="Number of process used in dataloader.")
    parser.add_argument('--use_pin_memory', action='store_true',
        help="Use pin memory in dataloader.")
    parser.add_argument('--report_ppl', action='store_true',
        help="Report perplexity score instead of loss.")
    parser.add_argument(
        "--ppl_output", type=str, help="File path to perplexity."
    )
    parser.add_argument("--existing_model", type=str, default=None,
        help="Path to existing model.")
    parser.add_argument("--device", type=str, default='cuda:0',
        help="Device to use.")
    parser.add_argument('--seed', type=int, default=42,
        help="random seed for initialization")
    args = parser.parse_args()

    return args


def evaluation(net, val_mlm_loader, val_contra_loader, args, device):
    """Evaluate model"""
    net.eval()
    val_contra_loss, val_mlm_loss = [], []
    logger.info(f"Start validation for MLM")
    # evaluate MLM objective
    if args.train_mlm:
        for batch in val_mlm_loader:
            with torch.no_grad():
                mlm_inputs = {'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
                norm_inputs = None
                if args.predict_ideology:
                    norm_inputs = ({'input_ids': batch[3].to(device),
                        'attention_mask': batch[1].to(device)}, batch[4].to(device))
                batch_norm_loss, batch_mlm_loss = net(norm_inputs, mlm_inputs)

                if args.predict_ideology:
                    batch_mlm_loss = 0.5*batch_norm_loss + 0.5*batch_mlm_loss
                if args.train_contrast:
                    batch_mlm_loss = batch_mlm_loss * args.loss_alpha
                if args.n_gpu > 1:
                    batch_mlm_loss = batch_mlm_loss.mean()  # mean() to average on multi-gpu parallel training

                val_mlm_loss.append(batch_mlm_loss.item())
        val_mlm_loss = torch.tensor(val_mlm_loss).mean()
    # evaluate triplet loss objectives
    if args.train_contrast:
        for batch in tqdm(val_contra_loader, desc="Validation"):
            with torch.no_grad():
                if args.use_ideo_loss:
                    contrast_alpha = args.contrast_alpha if args.use_story_loss and \
                        (args.contrast_loss == 'multi-negative' or
                        batch[3]['input_ids'].shape[0] > 0) else 1
                    batch_contra_loss = train_contrast_step(args, net, batch,
                        'ideo', contrast_alpha, device)
                    if batch_contra_loss is not None:
                        val_contra_loss.append(batch_contra_loss.item())
                if args.use_story_loss:
                    contrast_alpha = (1-args.contrast_alpha) if args.use_ideo_loss and \
                        batch_contra_loss is not None else 1
                    batch_contra_loss = train_contrast_step(args, net, batch,
                        'story', contrast_alpha, device)
                    if batch_contra_loss is not None:
                        val_contra_loss.append(batch_contra_loss.item())
        val_contra_loss = torch.tensor(val_contra_loss).mean()
    
    if args.report_ppl:
        # report perplexity
        ppl = math.exp(val_mlm_loss)
        logger.info(f"Perplexity: {ppl}")
        if os.path.isfile(args.ppl_output):
            with open(args.ppl_output, 'a') as fp:
                fp.write(f"{ppl}\n")
        else:
            with open(args.ppl_output, 'w') as fp:
                fp.write(f"{ppl}\n")
    else:
        logger.info(f"Total loss: {val_mlm_loss+val_contra_loss}\tContrastive loss: {val_contra_loss}\tMLM loss: {val_mlm_loss}")


def main():
    args = parse_args()
    # set random seed
    set_seed(args.seed)
    # set tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.report_ppl:
        assert args.ppl_output is not None
    device = torch.device(args.device)

    # Create model
    net = POLITICS(args.model_name, args.contrast_loss,
        args.per_gpu_contra_train_batch_size, ideo_margin=args.ideo_margin,
        story_margin=args.story_margin, ideo_tau=args.ideo_tau, story_tau=args.story_tau,
        num_article_limit=args.contra_num_article_limit, predict_ideology=args.predict_ideology)
    if args.existing_model is not None:
        net.lm.load_state_dict(torch.load(args.existing_model, map_location=device))
    net = net.to(device)
    
    mlm_train_batch_size = args.per_gpu_mlm_train_batch_size * args.n_gpu
    contra_train_batch_size = args.per_gpu_contra_train_batch_size * args.n_gpu

    # get data collator
    mlm_collator = MLMCollator(args.model_name, args.replace_entity, args.predict_ideology)

    if args.train_contrast:
        contra_collator = TupleCollator(args.contrast_loss, args.use_story_loss,
            args.n_gpu, contra_train_batch_size)

    # Get the datasets and dataloader
    if args.do_train:
        trn_mlm_data = MLMDataset(args.mlm_train_files, args.model_name,
            args.mask_entity, args.mask_sentiment, args.lexicon_dir)
        trn_mlm_loader = DataLoader(trn_mlm_data, batch_size=mlm_train_batch_size,
            shuffle=True, collate_fn=mlm_collator, num_workers=args.data_process_worker,
            pin_memory=args.use_pin_memory)
        if not args.train_mlm:
            # dummy dataloader
            trn_mlm_loader = range(len(trn_mlm_loader))
        if args.train_contrast:
            trn_contra_data = TupleDataset(args.contra_train_file, args.model_name)
            trn_contra_loader = DataLoader(trn_contra_data, batch_size=contra_train_batch_size,
                shuffle=True, collate_fn=contra_collator, num_workers=args.data_process_worker,
                pin_memory=args.use_pin_memory)
        
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if args.train_mlm:
            mlm_optimizer = AdamW(optimizer_grouped_parameters, lr=args.mlm_learning_rate, betas=(0.9, 0.98))
            # Scheduler and math around the number of training steps.
            if args.max_train_steps is not None:
                mlm_total_st = args.max_train_steps
            else:
                mlm_total_st = math.ceil(len(trn_mlm_loader) / args.mlm_gradient_accumulation_steps) * args.num_train_epochs
            mlm_scheduler = get_linear_schedule_with_warmup(mlm_optimizer, num_warmup_steps=int(mlm_total_st * 0.06),
                num_training_steps=mlm_total_st)
        
        if args.train_contrast:
            contra_optimizer = AdamW(optimizer_grouped_parameters, lr=args.contra_learning_rate, betas=(0.9, 0.98))
            if args.max_train_steps is not None:
                contra_total_st = args.max_train_steps
            else:
                contra_total_st = math.ceil(len(trn_contra_loader) / args.contra_gradient_accumulation_steps) * args.num_train_epochs
            contra_scheduler = get_linear_schedule_with_warmup(contra_optimizer, num_warmup_steps=int(contra_total_st * 0.06),
                num_training_steps=contra_total_st)
        
        # total training steps
        total_train_step = mlm_total_st if args.train_mlm else contra_total_st
        
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        if args.train_mlm:
            logger.info("========= MLM Task =========")
            logger.info(f"  Num examples = {len(trn_mlm_data)}")
            logger.info(f"  Instantaneous batch size per gpu = {args.per_gpu_mlm_train_batch_size}")
            total_batch_size = mlm_train_batch_size * args.mlm_gradient_accumulation_steps
            logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
            logger.info(f"  Total optimization steps = {mlm_total_st}")
        if args.train_contrast:
            logger.info("========= Contrastive Learning =========")
            logger.info(f"  Num examples = {len(trn_contra_data)}")
            logger.info(f"  Instantaneous batch size per gpu = {args.per_gpu_contra_train_batch_size}")
            total_batch_size = contra_train_batch_size * args.contra_gradient_accumulation_steps
            logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
            logger.info(f"  Total optimization steps = {contra_total_st}")

    # evaluation datasets
    if args.train_mlm:
        val_mlm_data = MLMDataset(args.mlm_val_files, args.model_name,
            args.mask_entity, args.mask_sentiment, args.lexicon_dir)
        val_batch_size = args.per_gpu_mlm_eval_batch_size * args.n_gpu
        drop_last = args.report_ppl
        val_mlm_loader = DataLoader(val_mlm_data, batch_size=val_batch_size,
            collate_fn=mlm_collator, num_workers=args.data_process_worker, drop_last=drop_last)
    else:
        val_mlm_loader = None

    if args.train_contrast:
        val_contra_data = TupleDataset(args.contra_val_file, args.model_name)
        val_batch_size = args.per_gpu_contra_eval_batch_size * args.n_gpu
        val_contra_loader = DataLoader(val_contra_data, batch_size=val_batch_size,
            collate_fn=contra_collator, num_workers=args.data_process_worker)
    else:
        val_contra_loader = None

    # DP
    if args.n_gpu > 1:
        net = torch.nn.DataParallel(net)

    # ========== Train ==========
    if args.do_train:
        step = 0
        global_step = 0
        mlm_loss, contra_loss = 0.0, 0.0
        net.zero_grad()

        progress_bar = tqdm(range(total_train_step), desc="Training")

        # create contrastive dataloader
        if args.train_contrast:
            contra_loader_iter = iter(trn_contra_loader)
        
        for epoch in range(args.num_train_epochs):
            if global_step >= total_train_step:
                break
            net.train()
            for batch in trn_mlm_loader:
                if args.train_mlm:
                    mlm_inputs = {'input_ids': batch[0].to(device),
                        'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
                    norm_inputs = None
                    if args.predict_ideology:
                        norm_inputs = ({'input_ids': batch[3].to(device),
                            'attention_mask': batch[1].to(device)}, batch[4].to(device))
                    
                    batch_norm_loss, batch_mlm_loss = net(norm_inputs, mlm_inputs)

                    if args.predict_ideology:
                        batch_mlm_loss = 0.5*batch_norm_loss + 0.5*batch_mlm_loss
                    if args.train_contrast:
                        batch_mlm_loss = batch_mlm_loss * args.loss_alpha
                    if args.n_gpu > 1:
                        batch_mlm_loss = batch_mlm_loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.mlm_gradient_accumulation_steps > 1:
                        batch_mlm_loss = batch_mlm_loss / args.mlm_gradient_accumulation_steps

                    batch_mlm_loss.backward()
                    # store mlm loss
                    mlm_loss += batch_mlm_loss.item()

                if (step + 1) % args.mlm_gradient_accumulation_steps == 0:
                    if args.train_mlm:
                        # update mlm
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                        mlm_optimizer.step()
                        mlm_scheduler.step()
                        net.zero_grad()

                    # train contrastive
                    if args.train_contrast:
                        for _ in range(args.contra_gradient_accumulation_steps):
                            # load batch
                            try:
                                batch = next(contra_loader_iter)
                            except StopIteration:
                                contra_loader_iter = iter(trn_contra_loader)
                                batch = next(contra_loader_iter)
                            if args.use_ideo_loss:
                                # ideology objective
                                contrast_alpha = args.contrast_alpha if args.use_story_loss and \
                                    (args.contrast_loss == 'multi-negative' or
                                    batch[3]['input_ids'].shape[0] > 0) else 1
                                batch_contra_loss = train_contrast_step(args, net, batch,
                                    'ideo', contrast_alpha, device)
                                if batch_contra_loss is not None:
                                    if args.contra_gradient_accumulation_steps > 1:
                                        batch_contra_loss = batch_contra_loss / args.contra_gradient_accumulation_steps
                                    batch_contra_loss.backward()
                                    contra_loss += batch_contra_loss.item()
                            if args.use_story_loss:
                                # story objective
                                contrast_alpha = (1-args.contrast_alpha) if args.use_ideo_loss and \
                                    batch_contra_loss is not None else 1
                                batch_contra_loss = train_contrast_step(args, net, batch,
                                    'story', contrast_alpha, device)
                                if batch_contra_loss is not None:
                                    if args.contra_gradient_accumulation_steps > 1:
                                        batch_contra_loss = batch_contra_loss / args.contra_gradient_accumulation_steps
                                    batch_contra_loss.backward()
                                    contra_loss += batch_contra_loss.item()

                        # update contra
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                        contra_optimizer.step()
                        contra_scheduler.step()
                        net.zero_grad()

                    # update global step
                    progress_bar.update(1)
                    global_step += 1
                    # log info
                    if global_step % args.logging_steps == 0:
                        logger.info(f"===== Global step: {global_step} =====\nTotal loss: {(contra_loss + mlm_loss)/args.logging_steps}")
                        logger.info(f"Contrastive loss: {contra_loss/args.logging_steps}\nMLM loss: {mlm_loss/args.logging_steps}")
                        if args.train_mlm:
                            logger.info(f"MLM learning rate: {mlm_scheduler.get_last_lr()[0]}")
                        if args.train_contrast:
                            logger.info(f"Contrastive learning rate: {contra_scheduler.get_last_lr()[0]}")
                        mlm_loss, contra_loss = 0.0, 0.0

                step += 1
                if global_step >= total_train_step:
                    break
            
            # store model
            base_name, _ = os.path.splitext(args.output_path)
            if args.n_gpu > 1:
                net.module.save_mlm_model(f"{base_name}_{epoch}.pt")
            else:
                net.save_mlm_model(f"{base_name}_{epoch}.pt")
            logger.info(f"=========== Epoch {epoch} ===========")
            evaluation(net, val_mlm_loader, val_contra_loader, args, device)
    # evaluation
    if args.do_eval:
        evaluation(net, val_mlm_loader, val_contra_loader, args, device)


def train_contrast_step(args, net, batch, contrast_type, contrast_alpha, device):
    """Contrastive training step"""
    batch_contra_loss = None
    mlm_inputs = None
    if args.contrast_loss == 'triplet':
        batch = batch[:3] if contrast_type == 'ideo' else batch[3:]
    if args.contrast_loss == 'triplet' and batch[0]['input_ids'].shape[0] > 0:
        norm_inputs = [{key: value.to(device) for key, value in each.items()} for each in batch]
        batch_contra_loss, _ = net(norm_inputs, mlm_inputs, contrast_type)
    elif args.contrast_loss == 'multi-negative':
        batch = [i.to(device) for i in batch]
        batch_contra_loss, _ = net(batch, mlm_inputs, contrast_type)
        batch_contra_loss = batch_contra_loss[batch_contra_loss != -100]
        if batch_contra_loss.shape[0] == 0:
            return None
    if batch_contra_loss is not None:
        if args.train_mlm:
            batch_contra_loss *= 1 - args.loss_alpha
        batch_contra_loss *= contrast_alpha
        if args.n_gpu > 1:
            batch_contra_loss = batch_contra_loss.mean()  # mean() to average on multi-gpu parallel training
        return batch_contra_loss
    else:
        return None


if __name__ == "__main__":
    main()