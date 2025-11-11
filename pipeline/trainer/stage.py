import os
from typing import Any, Dict
import torch
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from transformers import Adafactor

from llamo.modeling_llamo import LLaMoForConditionalGeneration
from llamo.tokenization_llamo import build_llamo_tokenizer
import time
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np

def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('<s>').__ne__, gt_tokens))
        gt_tokens = list(filter(('</s>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[INST]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[/INST]').__ne__, gt_tokens))



        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('<s>').__ne__, out_tokens))
        out_tokens = list(filter(('</s>').__ne__, out_tokens))
        out_tokens = list(filter(('[INST]').__ne__, out_tokens))
        out_tokens = list(filter(('[/INST]').__ne__, out_tokens))
        

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    model.load_state_dict(state_dict, strict=True)

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class LLaMoStage(pl.LightningModule):
    
    def __init__(self, args, config):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.llm_tune = args.llm_tune
        self.length_penalty= args.length_penalty
        self.gnn_path = args.gnn_path
        config.graph_config.gnn_path = self.gnn_path
        self.new_idx = 0

        self.mllm = LLaMoForConditionalGeneration(config)
        tokenizer_ckpt = config.lm_config.pretrained_tokenizer_name_or_path
        self.tokenizer = build_llamo_tokenizer(tokenizer_ckpt, config.num_query_tokens)


        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<s>'
        
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
        
        self.mllm.resize_token_embeddings(len(self.tokenizer))
        self.mllm.language_model.resize_token_embeddings(len(self.tokenizer))

        self.save_hyperparameters(args)

    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def on_test_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []
        self.list_tasks = []


    def save_predictions(self, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        keys = list(kwargs.keys())
        len_dump = len(kwargs[keys[0]])
        for k in keys:
            assert len(kwargs[k]) == len_dump

        filepath = os.path.join(self.logger.log_dir, f'dumps_rank_{rank}.json')
        # load the previous dumps from filepath
        if os.path.exists(filepath):
            # read jsonl file
            with open(filepath, 'r', encoding='utf8') as f:
                cumulative_dumps = json.load(f)
        else:
            cumulative_dumps = []

        for i in range(len_dump):
            line = {k: kwargs[k][i] for k in keys}
            cumulative_dumps.append(line)

        with open(filepath, 'w', encoding='utf8') as f:
            # save json file
            json.dump(cumulative_dumps, f, ensure_ascii=True, indent=4)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.mllm.eval()
        graph_values = batch['graph_values']
        input_ids = batch['input_ids']
        smiles = batch['smiles']
        seq_length = batch['seq_length']
        attention_mask = batch['attention_mask']
        label = batch['labels']

        ###============== Captioning Results ===================###
        prediction_ids = self.mllm.generate(
            graph_values=graph_values,
            input_ids=input_ids,
            smiles=smiles,
            seq_length=seq_length,
            attention_mask=attention_mask, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_new_tokens=self.max_len,
            min_length=self.min_len,
            length_penalty=self.length_penalty
        )
        
        predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        texts = self.tokenizer.batch_decode(label, skip_special_tokens=True)
        texts = [txt.strip() for txt in texts]

        self.list_predictions.append(predictions)
        self.list_targets.append(texts)

        if 'task' in batch.keys() and batch['task'] is not None:
            self.list_tasks.append(batch['task'])

        input_ids = batch['input_ids']
        input_ids = torch.where(input_ids < 0, 0, input_ids)
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        save_dict = {
            'tasks': batch['task'],
            'input_texts': input_texts,
            'targets': texts,
            'predictions': predictions,
        }
        self.save_predictions(**save_dict)

        return (predictions, texts)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.mllm.eval()

        graph_values = batch['graph_values']
        input_ids = batch['input_ids']
        smiles = batch['smiles']
        attention_mask = batch['attention_mask']
        label = batch['labels']
        tasks = batch['task']

        ###============== Captioning Results ===================###
        outputs = self.mllm.generate(
            graph_values=graph_values,
            input_ids=input_ids,
            smiles=smiles,
            attention_mask=attention_mask, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_new_tokens=self.max_len,
            min_length=self.min_len,
            length_penalty=self.length_penalty
        )

        prediction_ids = outputs.sequences

        predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        texts = self.tokenizer.batch_decode(label, skip_special_tokens=True)
        texts = [txt.strip() for txt in texts]

        self.list_predictions.append(predictions)
        self.list_targets.append(texts)

        if 'task' in batch.keys() and batch['task'] is not None:
            self.list_tasks.append(batch['task'])
        input_ids = batch['input_ids']
        input_ids = torch.where(input_ids < 0, 0, input_ids)
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        binary_classificaiton_probs = convert_logit2binary_prob(outputs.logits, self.tokenizer, tasks)

        save_dict = {
            'tasks': tasks,
            'input_texts': input_texts,
            'targets': texts,
            'predictions': predictions,
            'binary_classificaiton_probs': binary_classificaiton_probs,
        }

        self.save_predictions(**save_dict)


    
    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []
        self.list_tasks = []

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch['input_ids'].shape[0]
        
        ###============== Overall Loss ===================###
        outputs = self.mllm(**batch)
        loss = outputs.loss

        self.log("molecule loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=1)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=512)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--length_penalty', type=float, default=1.0)
        
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=10)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=5e-5, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=5e-7, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--stage_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        
        return parent_parser

def convert_logit2binary_prob(logits, tokenizer, tasks):
    classification_classes = {
    'bace',
    'smol-property_prediction-bbbp',
    'smol-property_prediction-clintox',
    'smol-property_prediction-hiv',
    'smol-property_prediction-sider',
    }
    classification_masks = []
    for task in tasks:
        if any(cls in task for cls in classification_classes):
            classification_masks.append(True)
        else:
            classification_masks.append(False)
    classification_masks = torch.tensor(classification_masks, dtype=torch.bool).unsqueeze(1)

    # positive token id
    positive_tokens = ["True", "true", "TRUE", "yes", "Yes", "YES"]
    positive_token_ids = [tokenizer.encode(token)[1] for token in positive_tokens]
    positive_token_ids = torch.tensor(positive_token_ids, dtype=torch.long)
    # negative token id
    negative_tokens = ["False", "false", "FALSE", "no", "No", "NO"]
    negative_token_ids = [tokenizer.encode(token)[1] for token in negative_tokens]
    negative_token_ids = torch.tensor(negative_token_ids, dtype=torch.long)

    probs = logits.softmax(-1)

    batch_size, sequence_length, vocab_size = probs.shape
    false_logits = torch.zeros(batch_size, 1)
    true_logits = torch.zeros(batch_size, 1)
    target_logits_index = torch.zeros((batch_size), dtype=torch.long)

    for i in range(batch_size):
        # inspect that prediction includes positive or negative tokens
        logits_i = logits[i, :, :]
        prediction_ids_i = logits_i.argmax(-1)

        if any(pos_id_i in prediction_ids_i for pos_id_i in positive_token_ids + negative_token_ids):
            # only get the first occurrence in the prediction_ids_i of among positive token
            for idx, pred_id_i in enumerate(prediction_ids_i):
                if pred_id_i in positive_token_ids + negative_token_ids:
                    target_logits_index[i] = idx
                    break
        else:
            target_logits_index[i] = 0

        false_logits[i] = probs[i, target_logits_index[i], negative_token_ids].sum()
        true_logits[i] = probs[i, target_logits_index[i], positive_token_ids].sum()

    total_probs = torch.cat(
        [false_logits, true_logits], dim=-1
    )
    total_probs = total_probs.softmax(-1)

    total_probs = torch.where(
        classification_masks,
        total_probs,
        torch.full_like(total_probs, -1),
    )

    total_probs = [p.tolist() for p in total_probs]
    return total_probs

