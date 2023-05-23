'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
import argparse
import yaml
import json
import random
from tqdm import tqdm
import numpy as np
import os
import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from fire import Fire
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from transformers import Seq2SeqTrainingArguments, TrainingArguments, IntervalStrategy, get_linear_schedule_with_warmup
from transformers.utils import logging

from model.RelationExt import RelationExt
from model.VAESampler import MetaVAE
from config import DataTrainingArguments, ModelArguments
from model.configs.config import get_config
from dataset import ExtDataTr, VAEData, get_dataloader, SingleExt
from cst_trainer import CustomTrainer as FTTrainer

from wrapper import Dataset as wr_Dataset
from wrapper import Sentence, Generator
from encoding import ExtractEncoder
from generation import LabelConstraint, TripletSearchDecoderPro
from utils import safe_divide, _get_learning_rate

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer(nn.Module):
    def __init__(self, train_args, vae_args, data_args,
                 num_training_steps_ext, num_training_steps_vae, search_threshold: float = -0.9906):
        super(Trainer, self).__init__()
        self.ext_model = RelationExt(train_args)
        self.vae = MetaVAE(vae_args)
        self.cuda = False
        self.max_length = data_args.max_target_length
        self.search_threshold = search_threshold

        if torch.cuda.is_available() and train_args.device.type == 'cuda':
            self.ext_model.cuda()
            self.vae.cuda()
            self.cuda = True

        self.optimizer_ext = torch.optim.AdamW(
            self.ext_model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay
        )
        self.optimizer_vae = torch.optim.AdamW(
            self.vae.parameters(), lr=vae_args.exp_params.LR, weight_decay=vae_args.exp_params.weight_decay
        )

        num_warmup_steps_ext = num_training_steps_ext * train_args.warmup_ratio
        num_warmup_steps_vae = num_training_steps_vae * train_args.warmup_ratio
        self.scheduler_ext = get_linear_schedule_with_warmup(self.optimizer_ext, num_warmup_steps=num_warmup_steps_ext,
                                                             num_training_steps=num_training_steps_ext)
        self.scheduler_vae = get_linear_schedule_with_warmup(self.optimizer_vae, num_warmup_steps=num_warmup_steps_vae,
                                                             num_training_steps=num_training_steps_vae)

    def train_step(self, data_batch, model_type='extraction', clip=1., gradient_accumulation_steps=1, k=3,
                   grad_step=True):  # todo: clip_grad_norm_, accumulate_step

        losses = []

        if model_type in ['extraction', 'both']:
            if model_type == 'extraction':
                data_ext = {key: value.cuda() if self.cuda else value for key, value in data_batch.items()}
                outputs_gen = self.ext_model(data_ext, model_type=model_type)
            else:
                data_ext, data_cnt, _ = {key: value.cuda() if self.cuda else value for key, value in
                                         data_batch[0].items()}, \
                                        {key: value.cuda() if self.cuda else value for key, value in
                                         data_batch[1].items()}, \
                                        data_batch[2]

                outputs_gen, logits_cnt = self.ext_model([data_ext, data_cnt], model_type='both')
                loss_cnt = self.ext_model.compute_cnt_loss(logits_cnt, temperature=1, k=k, kl_type='sum')

            # loss: extraction
            loss_gen = self.ext_model.compute_ce_loss(self.ext_model.model, data_ext, outputs_gen)
            if model_type == 'extraction':
                loss = loss_gen
            else:
                loss = loss_gen + 0.05* loss_cnt

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if grad_step:
                clip_grad_norm_(self.ext_model.parameters(), clip)
                self.optimizer_ext.step()
                self.scheduler_ext.step()
                self.ext_model.zero_grad()

            if model_type == 'extraction':
                losses.append({'loss': loss.detach().item(), 'loss_gen': loss_gen.detach().item(),
                               'loss_cnt': 0.})
            else:
                losses.append({'loss': loss.detach().item(), 'loss_gen': loss_gen.detach().item(),
                               'loss_cnt': loss_cnt.detach().item()})

        if model_type in ['vae', 'both']:
            if model_type == 'vae':
                input_vae, input_length, relations = [item.cuda() if self.cuda else item for item in data_batch]

            else:
                _, _, data_vae = data_batch
                data_vae = {key: value.cuda() if self.cuda else value
                            for key, value in data_vae.items()}

                input_vae = data_vae['input_vae']
                input_length = data_vae['input_length']
                relations = data_vae['relations']

            res_vae = self.vae(input_vae, input_length=input_length.squeeze(1), relations=relations)

            # loss: vae
            losses_vae = self.vae.loss_function(*res_vae, input_length=input_length.squeeze(1))
            loss_vae = losses_vae.pop('loss')  # check the pop mechanism
            loss_vae = loss_vae / gradient_accumulation_steps
            loss_vae.backward()

            if grad_step:
                clip_grad_norm_(self.vae.parameters(), clip)
                self.optimizer_vae.step()
                self.scheduler_vae.step()
                self.vae.zero_grad()

            losses_vae['loss'] = loss_vae.detach().item()
            losses.append(losses_vae)

        return losses

    def eval_step(self, data_batch, model_type='extraction', k=3):  # todo: evaluate ext result

        losses = []
        if model_type in ['extraction', 'both']:

            if model_type == 'extraction':
                data_ext = {key: value.cuda() if self.cuda else value for key, value in data_batch.items()}
                outputs_gen = self.ext_model(data_ext, model_type=model_type)
            else:
                data_ext, data_cnt, _ = {key: value.cuda() if self.cuda else value for key, value in
                                         data_batch[0].items()}, \
                                        {key: value.cuda() if self.cuda else value for key, value in
                                         data_batch[1].items()}, \
                                        data_batch[2]

                outputs_gen, logits_cnt = self.ext_model([data_ext, data_cnt], model_type=model_type)
                loss_cnt = self.ext_model.compute_cnt_loss(logits_cnt, temperature=1, k=k, kl_type='sum')

            # loss: extraction
            loss_gen = self.ext_model.compute_ce_loss(self.ext_model.model, data_ext, outputs_gen)

            if model_type == 'extraction':
                loss = loss_gen
                losses.append({'loss': loss.detach().item(),
                               'loss_gen': loss_gen.detach().item(),
                               'loss_cnt': 0.})
            else:
                loss = loss_gen + 0.05* loss_cnt

                losses.append({'loss': loss.detach().item(),
                               'loss_gen': loss_gen.detach().item(),
                               'loss_cnt': loss_cnt.detach().item()})

        if model_type in ['vae', 'both']:
            if model_type == 'vae':
                input_vae, input_length, relations = [item.cuda() if self.cuda else item for item in data_batch]
            else:
                _, _, data_vae = data_batch
                data_vae = {key: value.cuda() if self.cuda else value
                            for key, value in data_vae.items()}

                input_vae = data_vae['input_vae']
                input_length = data_vae['input_length']
                relations = data_vae['relations']

            res_vae = self.vae(input_vae, input_length=input_length.squeeze(1), relations=relations)

            # loss: vae
            losses_vae = self.vae.loss_function(*res_vae, input_length=input_length.squeeze(1))
            loss_vae = losses_vae['loss']

            losses_vae['loss'] = loss_vae.detach().item()
            losses.append(losses_vae)

        return losses

    def predict(self, path_in: str, path_out: str, tokenizer: Any, batch_size: int = 32,
                model_type: str = 'extraction', use_label_constraint: bool = True, use_mask: bool = False):

        if model_type in ['extraction', 'both']:
            data = wr_Dataset.load(path_in)
            texts = [s.text for s in data.sents]
            encoder = ExtractEncoder()
            self.ext_model.tokenizer = tokenizer
            constraint = LabelConstraint(labels=data.get_labels(), tokenizer=tokenizer)
            sents = []

            # for data_batch in tqdm(dataset, desc='predicting'):
            for i in tqdm(range(0, len(texts), batch_size), desc='predicting'):
                batch = texts[i: i + batch_size]
                if use_mask:
                    x = [encoder.encode_x_(t) for t in batch]
                else:
                    x = [encoder.encode_x(t) for t in batch]

                # data_ext = {key: value.cuda() if self.cuda else value
                #             for key, value in data_batch.items() if key not in ['labels', 'decoder_input_ids']}
                outputs = self.ext_model.run(x, tokenizer=tokenizer,
                                             max_length=self.max_length,
                                             save_scores=use_label_constraint,
                                             num_return=1,
                                             num_beams=1,
                                             do_sample=False)

                for j, raw in enumerate(outputs):
                    # x = self.ext_model.decode(data_ext['input_ids'], tokenizer=tokenizer)
                    # x = self.ext_model.decode(x[i], tokenizer=tokenizer)
                    if use_mask:
                        triplet = encoder.safe_decode_(x[j], y=raw)
                    else:
                        triplet = encoder.safe_decode(x[j], y=raw)
                    if use_label_constraint:
                        assert self.ext_model.scores is not None
                        triplet = constraint.run(triplet, self.ext_model.scores[j])
                    sents.append(Sentence(triplets=[triplet]))

            wr_Dataset(sents=sents).save(path_out)

    def predict_single(self, path_in: str, path_out: str, tokenizer: Any, use_mask: bool = False):

        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")
        print(dict(predict_single=locals()))
        data = wr_Dataset.load(path_in)
        encoder = ExtractEncoder()
        self.ext_model.tokenizer = tokenizer
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=tokenizer)
        searcher = TripletSearchDecoderPro(
            gen=self.ext_model, encoder=encoder, constraint=constraint, top_k=2
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text, use_mask=use_mask))
            for s in tqdm(data.sents)
            # for idx, s in enumerate(tqdm(data.sents)) if idx==0
        ]
        wr_Dataset(sents=sents).save(path_raw)
        for s in sents:
            scores = [t.score for t in s.triplets]
            top_idx = np.argmax(scores)
            s.triplets = [s.triplets[top_idx]]
        wr_Dataset(sents=sents).save(path_out)

    def predict_multi(self, path_in: str, path_out: str, tokenizer: Any, use_mask: bool = False):

        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")
        print(dict(predict_multi=locals()))
        data = wr_Dataset.load(path_in)
        encoder = ExtractEncoder()
        self.ext_model.tokenizer = tokenizer
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=tokenizer)
        searcher = TripletSearchDecoderPro(
            gen=self.ext_model, encoder=encoder, constraint=constraint
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text, use_mask=use_mask))
            for s in tqdm(data.sents)
            # for idx, s in enumerate(tqdm(data.sents)) if idx==0
        ]
        wr_Dataset(sents=sents).save(path_raw)
        for s in sents:
            s.triplets = [t for t in s.triplets if t.score > self.search_threshold]
        wr_Dataset(sents=sents).save(path_out)

    def finetune_with_sampler(self, dataset, val_dataset, training_args,
                              logger, path, k=3):

        logger.info("*** Model initialization ***")
        trainer = FTTrainer(
            model=self.ext_model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            data_collator=dataset.collate_fn,
            vae_sampler=self.vae,
            k=k,
            path=path,
        )

        logger.info("*** Train ***")

        train_model(trainer, dataset)

        logger.info("*** Evaluate ***")

        eval_model(trainer, val_dataset)

    def train_no_sampler(self, epoch, logger, dataloaders, path, accumulate_gr=1, model_type='extraction', save_each=False):
        path_model = Path(path) / f"{model_type}.pt"
        # if not Path(path_model).exists():
        #     Path(path_model).parent.mkdir(exist_ok=True, parents=True)
        dataloader_tr, dataloader_dev = dataloaders
        # eval_loss = []
        best_loss = 1e+6
        if model_type == 'extraction':
            model = self.ext_model
        else:
            model = self.vae

        model.zero_grad()
        for i in tqdm(range(epoch), desc='train ' + model_type, leave=True):

            model.train()
            results = {}
            for idx, dataset in tqdm(enumerate(dataloader_tr), desc='iteration', leave=True, position=0):
                grad_step = ((idx + 1) % accumulate_gr == 0) or (idx + 1 == len(dataloader_tr))
                res = self.train_step(dataset, gradient_accumulation_steps=accumulate_gr,
                                      model_type=model_type, grad_step=grad_step)
                results = Trainer._summary(results, res[0])

            logs = {k: v / len(dataloader_tr) for k, v in results.items()}
            if model_type == 'extraction':

                logs.update({'lr_ext': _get_learning_rate(self.scheduler_ext)})
            else:
                logs.update({'lr_vae': _get_learning_rate(self.scheduler_vae)})
            logger.info(logs)

            with torch.no_grad():
                model.eval()
                results = {}
                for dataset in tqdm(dataloader_dev, desc='iteration', leave=True, position=1):
                    res = self.eval_step(dataset, model_type=model_type)
                    results = Trainer._summary(results, res[0])

                    # eval_loss.append(res[0]['loss'])

                logger.info({k: v / len(dataloader_dev) for k, v in results.items()})
                if results['loss'] / len(dataloader_dev) <= best_loss:
                    best_loss = results['loss'] / len(dataloader_dev)
                    model_state = model.state_dict()
                    torch.save(model_state, path_model)

                if save_each:
                    model_state = model.state_dict()
                    path_model_ = Path(path) / f"{model_type}_{i}.pt"
                    torch.save(model_state, path_model_)

            # eval_loss = []
        # save_model(model_type)

    @staticmethod
    def _summary(results, res):
        for k, v in res.items():
            if k not in results:
                results[k] = v
            else:
                results[k] += v
        return results

    def evaluate(self):
        return

    def load_model(self, model_ext, model_vae):
        self.ext_model.load_state_dict(torch.load(model_ext))
        self.vae.load_state_dict(torch.load(model_vae))

    def load_model_(self, model_ext, model_vae):
        self.ext_model.model.load_state_dict(torch.load(model_ext))
        self.vae.load_state_dict(torch.load(model_vae))

    @staticmethod
    def score(path_pred: str, path_gold: str) -> dict:
        pred = wr_Dataset.load(path_pred)
        gold = wr_Dataset.load(path_gold)
        assert len(pred.sents) == len(gold.sents)
        num_pred = 0
        num_gold = 0
        num_correct = 0

        for i in range(len(gold.sents)):
            num_pred += len(pred.sents[i].triplets)
            num_gold += len(gold.sents[i].triplets)
            for p in pred.sents[i].triplets:
                for g in gold.sents[i].triplets:
                    if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        info = dict(
            path_pred=path_pred,
            path_gold=path_gold,
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
        )
        return info


# train model with trainer
def train_model(trainer, train_dataset):

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


# evaluate model with trainer
def eval_model(trainer, val_dataset):
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(val_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def training(save_dir, path_model, data_name, split, logger):
    ext_save_dir = str(Path(save_dir) / "extractor")
    model_args, train_args, data_args, vae_args = get_args(ext_save_dir, do_pretrain=True)
    # print("model_args: ", model_args)
    # print("train_args: ", train_args)
    # print("data_args: ", data_args)
    train_vae = VAEData(path=save_dir, train_args=train_args, name='train', data_name=data_name, split=split)
    dev_vae = VAEData(path=save_dir, train_args=train_args, name='dev', data_name=data_name, split=split)
    vae_dataloader_tr, _ = get_dataloader(train_vae, model_type='single_vae', bz=train_args.per_device_train_batch_size)
    vae_dataloader_dev, _ = get_dataloader(dev_vae, model_type='single_vae', bz=train_args.per_device_train_batch_size)

    ext_dataloader_tr, _ = get_dataloader(['train', save_dir, model_args, train_args, data_args, vae_args,
                                           data_name, split],
                                          model_type='single_ext', bz=train_args.per_device_train_batch_size)
    ext_dataloader_dev, _ = get_dataloader(['dev', save_dir, model_args, train_args, data_args, vae_args,
                                            data_name, split],
                                           model_type='single_ext', bz=train_args.per_device_train_batch_size)
    len_dataloader = len(ext_dataloader_tr)
    num_update_steps_per_epoch = len_dataloader // train_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_training_steps_ext = math.ceil(train_args.num_train_epochs * num_update_steps_per_epoch)
    num_training_steps_vae = math.ceil(vae_args.trainer_params.max_epochs * num_update_steps_per_epoch)
    vae_num_class = len(train_vae.rel_vocab)
    vae_vocab_size = len(train_vae.vae_vocab.word2index)
    vae_args.model_params.num_classes = vae_num_class
    vae_args.model_params.vocab_size = vae_vocab_size
    trainer = Trainer(train_args, vae_args, data_args, num_training_steps_ext, num_training_steps_vae)

    # path_ = str(Path(save_dir) / "runs")

    if not Path(path_model).exists():
        Path(path_model).mkdir(exist_ok=True, parents=True)

    trainer.train_no_sampler(vae_args.trainer_params.max_epochs, logger, [vae_dataloader_tr, vae_dataloader_dev],
                             path_model,
                             model_type='vae', accumulate_gr=train_args.gradient_accumulation_steps)
    # trainer.train_no_sampler(train_args.num_train_epochs, logger, [ext_dataloader_tr, ext_dataloader_dev], path_model,
    #                          model_type='extraction', accumulate_gr=train_args.gradient_accumulation_steps)


def finetuning(save_dir, path_model, data_name, split, logger, last=False, k=3):
    ext_save_dir = str(Path(save_dir) / "extractor")
    model_args, train_args, data_args, vae_args = get_args(ext_save_dir, do_pretrain=False)
    # print("finetune_args: ", train_args)

    train_vae = VAEData(path=save_dir, train_args=train_args, name='synthetic', data_name=data_name, split=split)
    dev_vae = VAEData(path=save_dir, train_args=train_args, name='dev', data_name=data_name, split=split)
    vae_dataloader_tr, _ = get_dataloader(train_vae, model_type='single_vae', bz=train_args.per_device_train_batch_size)
    vae_dataloader_dev, _ = get_dataloader(dev_vae, model_type='single_vae', bz=train_args.per_device_train_batch_size)

    # ext_dataloader_dev, _ = get_dataloader(['dev', save_dir, model_args, train_args, data_args, vae_args,
    #                                         data_name, split],
    #                                        model_type='single_ext', bz=train_args.per_device_train_batch_size)
    # dev_data = SingleExt('dev', save_dir, model_args, train_args, data_args, vae_args,
    #                      data_name=data_name, split=split)
    dev_data = ExtDataTr('dev', save_dir, model_args, train_args, data_args, vae_args,
                         data_name=data_name, split=split)
    # dev_data.initialize()

    len_dataloader = len(vae_dataloader_tr)
    num_update_steps_per_epoch = len_dataloader // train_args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_training_steps = math.ceil(train_args.num_train_epochs * num_update_steps_per_epoch)
    # num_training_steps = len(ext_dataloader_tr) * train_args.num_train_epochs

    train_data = ExtDataTr('synthetic', save_dir, model_args, train_args, data_args, vae_args,
                         data_name=data_name, split=split)
    vae_num_class = len(train_data.rel_vocab)
    vae_vocab_size = len(train_data.vae_vocab.word2index)
    vae_args.model_params.num_classes = vae_num_class
    vae_args.model_params.vocab_size = vae_vocab_size
    trainer = Trainer(train_args, vae_args, data_args, num_training_steps, num_training_steps)

    # path_ = str(Path(save_dir) / "runs")
    path_model_ext = Path(path_model) / "extraction.pt"
    # path_model_ext = Path(path_model) / "pytorch_model.bin"
    path_model_vae = Path(path_model) / "vae.pt"
    if Path(str(path_model_ext)).exists() and Path(str(path_model_vae)).exists():
        trainer.load_model(path_model_ext, path_model_vae)
        print("using trained model")
    else:
        print("using pretrained model from huggingface")

    trainer.train_no_sampler(train_args.num_train_epochs, logger, [vae_dataloader_tr, vae_dataloader_dev],
                             path_model,
                             model_type='vae', accumulate_gr=train_args.gradient_accumulation_steps, save_each=True)

    del vae_dataloader_dev, vae_dataloader_tr

    trainer.finetune_with_sampler(train_data, dev_data, train_args, logger,
                                  path_model, k=k)


def get_args(ext_save_dir: str, do_pretrain: bool = False):
    model_name: str = "facebook/bart-base"
    max_source_length: int = 128
    max_target_length: int = 128
    encoder_name: str = "new_generate"
    pipe_name: str = "summarization"

    data_dir = str(Path(ext_save_dir) / "data")
    train_file = str(Path(data_dir) / f"{'train'}.json")
    validation_file = str(Path(data_dir) / f"{'dev'}.json")
    kwargs = {}

    data_args = DataTrainingArguments(
        train_file=train_file,
        validation_file=validation_file,
        overwrite_cache=True,
        max_target_length=max_target_length,
        max_source_length=max_source_length,
        **kwargs,
    )

    extarg = ExtractorArg(
        model_dir=str(Path(ext_save_dir) / "model"),
        data_dir=data_dir,
        do_pretrain=do_pretrain)
    train_args = extarg.get_train_args(do_eval=True)

    kwargs = {
        k: v for k, v in train_args.to_dict().items() if not k.startswith("_") and
                                                         k not in ['log_level', 'log_level_replica']
    }

    train_args = Seq2SeqTrainingArguments(**kwargs)
    model_args = ModelArguments(
        model_name_or_path=model_name
    )

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='./model/configs/cvae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            vae_args = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    vae_args = get_config(vae_args)

    return model_args, train_args, data_args, vae_args


# from transformers.utils import logging
# logger = logging.get_logger(__name__)
def gen_synthetic(
        save_dir: str,
        path_train: str,
        path_dev: str,
        path_test: str,
        tail: bool = True

):
    generator = Generator(
        load_dir="gpt2",  # str(Path(save_dir) / "generator" / "model") use this for load trained generator
        save_dir=str(Path(save_dir) / "generator"),
    )

    generator.fit(path_train, path_dev, tail=tail)
    path_synthetic = str(Path(save_dir) / "synthetic.jsonl")

    if tail:
        labels_dev = wr_Dataset.load(path_dev).get_symprompt()
        labels_test = wr_Dataset.load(path_test).get_symprompt()
        labels = wr_Dataset.combine_label(labels_dev, labels_test)
        generator.generate_(labels, path_out=path_synthetic)
    else:
        labels_dev = wr_Dataset.load(path_dev).get_labels()
        labels_test = wr_Dataset.load(path_test).get_labels()
        generator.generate(labels_dev + labels_test, path_out=path_synthetic)


def main(
        path_train: str,
        path_dev: str,
        path_test: str,
        save_dir: str,
        path_model: str,
        data_name: str,
        split: str,
        last: bool = False,
):
    # gen_synthetic(save_dir, path_train, path_dev, path_test)
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")

    training(save_dir, path_model, data_name, split, logger)
    finetuning(save_dir, path_model, data_name, split, logger, last=last)
    print("done!")
    # PersonaData(_config)_config containig ext_save_dir


def run_eval(save_dir: str, path_model: str, path_test: str, data_name: str, split: str, mode: str,
             last: bool = False, limit: int = 0):
    print(dict(run_eval=locals()))
    ext_save_dir = str(Path(save_dir) / "extractor")
    model_args, train_args, data_args, vae_args = get_args(ext_save_dir)
    data = wr_Dataset.load(path_test)
    # model = Extractor(load_dir=str(Path(path_model) / "model"), save_dir=path_model)
    data_dir = str(Path(ext_save_dir) / "data")
    path_vocab = Path(data_dir) / "vae_vocab.json"
    path_rel = Path(data_dir) / "rel_vocab.json"
    if Path(str(path_vocab)).exists():
        with open(path_vocab, 'r') as f:
            vae_vocab = json.load(f)
        vae_vocab_size = len(vae_vocab)
        vae_args.model_params.vocab_size = vae_vocab_size
    if Path(str(path_rel)).exists():
        with open(path_rel, 'r') as f:
            vae_rel = json.load(f)
        vae_num_class = len(vae_rel)
        vae_args.model_params.num_classes = vae_num_class

    trainer = Trainer(train_args, vae_args, data_args, 1, 1)
    if not last:
        # path_model_ext = Path(path_model) / "extraction_final.pt"
        path_model_ext = Path(path_model) / "pytorch_model_trn_850.bin"
    else:
        path_model_ext = Path(path_model) / "extraction_last.pt"
    path_model_vae = Path(path_model) / "vae_final.pt"
    if Path(str(path_model_ext)).exists() and Path(str(path_model_vae)).exists():
        trainer.load_model(path_model_ext, path_model_vae)
    else:
        print("using pretrained model from huggingface")

    if mode == "single" or mode == "trsingle":
        data.sents = [s for s in data.sents if len(s.triplets) == 1]
    elif mode == "multi":
        data.sents = [s for s in data.sents if len(s.triplets) > 1]
    else:
        raise ValueError(f"mode must be single or multi")

    if limit > 0:
        random.seed(0)
        random.shuffle(data.sents)
        data.sents = data.sents[:limit]

    path_in = str(Path(path_model) / f"pred_in_{mode}.jsonl")
    path_out = str(Path(path_model) / f"pred_out_{mode}.jsonl")
    data.save(path_in)

    test_data, tokenizer = get_dataloader(['test', save_dir, model_args, train_args, data_args, vae_args,
                                           data_name, split],
                                          model_type='single_ext', shuffle=False)
    if mode == "single":
        trainer.predict(path_in=path_in, path_out=path_out,
                        tokenizer=tokenizer, batch_size=train_args.per_device_train_batch_size)
    elif mode == "trsingle":
        trainer.predict_single(path_in=path_in, path_out=path_out,
                              tokenizer=tokenizer)
    else:
        trainer.predict_multi(path_in=path_in, path_out=path_out, tokenizer=tokenizer)

    results = trainer.score(path_pred=path_out, path_gold=path_in)
    path_results = str(Path(path_model) / f"results_{mode}.json")
    results.update(mode=mode, limit=limit, path_results=path_results)
    print(json.dumps(results, indent=2))
    with open(path_results, "w") as f:
        json.dump(results, f, indent=2)


class ExtractorArg(BaseModel):
    model_dir: str
    do_pretrain: bool
    batch_size: int = 8  # 8 # 32  # 64
    grad_accumulation: int = 4  # 16  # 1  # 2
    random_seed: int = 42  # 42
    warmup_ratio: float = 0.2
    lr_pretrain: float = 3e-5  # 3e-4
    lr_finetune: float = 6e-6  # 8e-6  # 6e-6  # 3e-5
    epochs_pretrain: int = 5  # 3
    epochs_finetune: int = 5
    label_smoothing_factor: float = 0.
    train_fp16: bool = True  # False  # True

    def get_train_args(self, do_eval: bool) -> TrainingArguments:
        return TrainingArguments(
            seed=self.random_seed,
            do_train=True,
            do_eval=do_eval or None,  # False still becomes True after parsing
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accumulation,
            warmup_ratio=self.warmup_ratio,
            output_dir=self.model_dir,
            save_strategy=IntervalStrategy.EPOCH,
            evaluation_strategy=IntervalStrategy.EPOCH
            if do_eval
            else IntervalStrategy.NO,
            learning_rate=self.get_lr(),
            num_train_epochs=self.get_epochs(),
            load_best_model_at_end=True,
            fp16=self.train_fp16,
            label_smoothing_factor=self.label_smoothing_factor,
        )

    def get_lr(self) -> float:
        return self.lr_pretrain if self.do_pretrain else self.lr_finetune

    def get_epochs(self) -> int:
        return self.epochs_pretrain if self.do_pretrain else self.epochs_finetune


if __name__ == "__main__":
    # main(save_dir="outputs/wrapper/u2t_map_all/unseen_10_seed_0",
    #     path_model="outputs/wrapper/u2t_map_all/unseen_10_seed_0/runs",
    #     split="unseen_10_seed_0/")
    # run_eval(save_dir="outputs/wrapper/u2t_map_all/unseen_10_seed_0",
    #          path_model="outputs/wrapper/u2t_map_all/unseen_10_seed_0/runs",
    #          path_test="outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/test.jsonl",
    #          split="unseen_10_seed_0/",
    #          mode='single')
    # run_eval(save_dir="outputs/wrapper/u2t_map_all/unseen_10_seed_0",
    #          path_model="outputs/wrapper/u2t_map_all/unseen_10_seed_0/runs",
    #          path_test="outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/test.jsonl",
    #          split='unseen_10_seed_0/',
    #          mode='multi')
    occp_tensor = torch.randn([2000, 1000, 1000]).cuda()
    del occp_tensor
    num_test_labels = [10]  # [5, 10, 15]
    seeds = [0]  # [0, 1, 2, 3, 4]
    for n in num_test_labels:
        for s in seeds:
            split_ = f"unseen_{n}_seed_{s}"
            print("processing split:", split_)
            # split_ = 'unseen_10_seed_0'
            run_name = '/runs'
            data_name = 'u2t_map_all'
            save_dir = "outputs/wrapper/" + data_name + "/" + split_
            path_model = save_dir + run_name
            path_train = "outputs/data/splits/zero_rte/" + data_name + "/" + split_ + "/train.jsonl"
            path_dev = "outputs/data/splits/zero_rte/" + data_name + "/" + split_ + "/dev.jsonl"
            path_test = "outputs/data/splits/zero_rte/" + data_name + "/" + split_ + "/test.jsonl"
            split = split_ + "/"
            # main(
            #     path_train=path_train,
            #     path_dev=path_dev,
            #     path_test=path_test,
            #     save_dir=save_dir,
            #     path_model=path_model,
            #     data_name=data_name,
            #     split=split,
            #     last=False)
            run_eval(save_dir=save_dir,
                     path_model=path_model,
                     path_test=path_test,
                     split=split,
                     data_name=data_name,
                     mode='single',
                     last=False)
            # run_eval(save_dir=save_dir,
            #          path_model=path_model,
            #          path_test=path_test,
            #          data_name=data_name,
            #          split=split,
            #          mode='trsingle',
            #          last=False)
            # run_eval(save_dir=save_dir,
            #          path_model=path_model,
            #          path_test=path_test,
            #          split=split,
            #          mode='multi')
