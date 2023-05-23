'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
'''
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Any
from datasets import load_dataset
from transformers import AutoTokenizer

from encoding import ExtractEncoder
from wrapper import Dataset as wr_Dataset


class Vocabulary:

    def __init__(self):

        PAD_token = 0  # Used for padding short sentences
        SOS_token = 1  # Start-of-sentence token
        EOS_token = 2  # End-of-sentence token

        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
        self.word2count = {"PAD": 0, "SOS": 0, "EOS": 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def update_vocab(self):
        for key in self.word2index.keys():
            self.index2word[self.word2index[key]] = key

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def to_words(self, indices):
        return [self.to_word(index) for index in indices]

    def to_indices(self, words):
        return [self.to_index(word) for word in words.split(' ')] + [self.word2index["EOS"]]


class VAEData(Dataset):
    def __init__(self, datas=None, vocab=None, path=None, train_args=None,
                 name='train', data_name='u2t_map_all', split='unseen_10_seed_0/'):
        if datas is None:
            path_in = "outputs/data/splits/zero_rte/" + data_name + "/" + split
            self.save_dir = str(Path(path) / "extractor")
            self.path = path
            self.random_seed = train_args.seed
            self.name = name
            self.gen_init_inputs(path_in)
            self.get_vocab(path_in)

            self.relations_idx = [self.rel_vocab[rel] for rel in self.relations]
        else:
            [self.vae_inputs, self.relations_idx], self.vae_vocab = datas, vocab

        self.sents = []

        for sent in self.vae_inputs:
            self.sents.append(self.vae_vocab.to_indices(sent))

        self.len = len(self.sents)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return torch.LongTensor(self.sents[index]), \
               torch.LongTensor([len(self.sents[index])]),\
                torch.LongTensor([self.relations_idx[index]])
               # torch.LongTensor([self.rel_vocab[self.relations[index]]]), \


    def collate_fn(self, data):
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'
        data.sort(key=lambda x: x[1], reverse=True)
        # len_sorted, indices = torch.sort(data[2], descending=True)
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[idx], batch_first=True) for idx in dat]

    def tokenize(self, inputs, tokenizer=None):

        inps, rel_ids = inputs
        model_inputs = {"input_ids": [], "relation_ids": rel_ids}
        for inp in inps:
            model_inputs['input_ids'].append(tokenizer.to_indices(inp))

        return model_inputs

    def preprocess_fun(self):

        self.vae_inputs = []
        self.vae_rels = []

        for idx, line in enumerate(self.data[self.name]['text']):

            vae_inp = line.split("Context : ")[-1]
            self.vae_inputs.append(vae_inp)

            true_rel = self.relations[idx]
            true_rel_idx = self.rel_vocab[true_rel]
            self.vae_rels.append(true_rel_idx)

        self.vae_inputs_ids = self.tokenize([self.vae_inputs, self.vae_rels], tokenizer=self.vae_vocab)

    @classmethod
    def load(cls, path: str, name: str):
        data_files = {}
        data_files[name] = path
        extension = path.split(".")[-1]
        sents = load_dataset(
            extension, data_files=data_files, cache_dir=None
        )
        return sents

    def write_data(self, data: wr_Dataset, name: str) -> str:
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = ExtractEncoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(self.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))

        return str(path_out)

    def get_data(self, path_in: str, name: str):
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / f"{name}.json"
        if Path(str(path_out)).exists():
            dataset = self.load(str(path_out), name)
        else:
            path_in = Path(path_in) / f"{name}.jsonl"
            data = wr_Dataset.load(str(path_in))
            path_out = self.write_data(data, name)
            dataset = self.load(path_out, name)

        return dataset

    def decode(self, summary: str) -> Tuple[Any, Any, Any]:
        front, back = summary.split(" , Relation : ")
        relation = back.split(' .')[0]
        front_, tail = front.split(" , Tail Entity :")
        head = front_.split("Head Entity : ")[-1]
        return head, tail, relation

    def get_vocab(self, path_in: str):

        self.vae_vocab = Vocabulary()
        self.rev_rel_vocab = {}
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / "vae_vocab.json"
        path_out_rel = Path(data_dir) / "rel_vocab.json"
        if Path(str(path_out)).exists():
            with open(path_out, 'r') as f:
                self.vae_vocab.word2index = json.load(f)
            self.vae_vocab.update_vocab()

            with open(path_out_rel, 'r') as f:
                self.rel_vocab = json.load(f)

        else:
            self.rel_vocab = {}
            for name in ['train', 'dev', 'synthetic']:
                if name == 'synthetic':
                    dataset = self.get_data(self.path, name)
                else:
                    dataset = self.get_data(path_in, name)
                for idx, line in tqdm(enumerate(dataset[name]['text']), desc='get_vae_vocab', leave=True):
                    head, tail, relation = self.decode(dataset[name]['summary'][idx])
                    if relation not in self.rel_vocab:
                        self.rel_vocab[relation] = len(self.rel_vocab)
                    vae_inp = line.split("Context : ")[-1]
                    self.vae_vocab.add_sentence(vae_inp)
            with open(path_out, 'w') as f:
                json.dump(self.vae_vocab.word2index, f)
            with open(path_out_rel, 'w') as f:
                json.dump(self.rel_vocab, f)

        # generate reverse_rel_vocab: index to relation
        for key in self.rel_vocab.keys():
            self.rev_rel_vocab[self.rel_vocab[key]] = key

        print("number of relations : ", len(self.rel_vocab))
        print("number of vae vocabularies : ", len(self.vae_vocab.word2index))

    def gen_init_inputs(self, path_in: str):

        self.vae_inputs = []
        self.relations = []
        if self.name == 'synthetic':
            self.data = self.get_data(self.path, self.name)
        else:
            self.data = self.get_data(path_in, self.name)
        for idx, line in tqdm(enumerate(self.data[self.name]['summary']), desc='gen_init...', leave=True):
            head, tail, relation = self.decode(line)
            self.relations.append(relation)

            context = self.data[self.name]['text'][idx]
            vae_inp = context.split("Context : ")[-1]
            self.vae_inputs.append(vae_inp)


class SingleExt(Dataset):
    def __init__(self, name, path, model_args, train_args, data_args, vae_args,
                 data_name='u2t_map_all', split='unseen_10_seed_0/'):
        self.path_in = "outputs/data/splits/zero_rte/" + data_name + "/" + split
        self.path = path
        self.name = name
        self.save_dir = str(Path(path) / "extractor")
        self.random_seed = train_args.seed
        self.batch_sz = train_args.per_device_train_batch_size
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.padding = "max_length" if data_args.pad_to_max_length else True #False
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss
        self.decoder_start_token_id = 2  # same as BartConfig
        if torch.cuda.is_available() and len(vae_args.trainer_params.gpus) != 0:
            self.cuda = True
        else:
            self.cuda = False

        tokenizer_kwargs = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
                        model_args.tokenizer_name
                        if model_args.tokenizer_name
                        else model_args.model_name_or_path,
                        cache_dir=model_args.cache_dir,
                        use_fast=model_args.use_fast_tokenizer,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        **tokenizer_kwargs, )

    def __getitem__(self, index):

        return self.gen_inputs_idx['input_ids'][index], \
               self.gen_inputs_idx['attention_mask'][index], \
               self.gen_inputs_idx['labels'][index], \
               self.gen_inputs_idx['decoder_input_ids'][index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):

        dat = pd.DataFrame(data)

        ext_inp = pad_sequence(dat[0], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ext_att = pad_sequence(dat[1], batch_first=True, padding_value=0)
        data_ext = {"input_ids": ext_inp, "attention_mask": ext_att,
                    "labels": pad_sequence(dat[2], batch_first=True, padding_value=-100),
                    "decoder_input_ids": pad_sequence(dat[3], batch_first=True, padding_value=self.tokenizer.pad_token_id)}

        return data_ext

    def initialize(self):
        if self.name == 'synthetic':
            self.data = self.get_data(self.path, self.name)
        else:
            self.data = self.get_data(self.path_in, self.name)
        self.gen_init_inputs()
        self.preprocess_fun()

        self.len = len(self.data[self.name]['text'])
        print('\n dataloader initialized!')

    def compute_dis(self, mus, logvars):
        raise NotImplementedError

    def vae_sampling(self, model:Any, k=1):
        raise NotImplementedError

    def tokenize(self, inputs, tokenizer=None, targets=None):

        if tokenizer is not None:
            inps, rel_ids = inputs
            model_inputs = {"input_ids":[], "relation_ids":rel_ids}
            for inp in inps:
                model_inputs['input_ids'].append(tokenizer.to_indices(inp))
        else:
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_source_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt"
            )
            if targets is not None:
                # Setup the tokenizer for targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets, max_length=self.max_target_length, padding=self.padding, truncation=True,
                        return_tensors="pt"
                    )

                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                        for label in labels["input_ids"]
                    ]

                model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_fun(self):

        self.gen_inputs_idx = self.tokenize(self.gen_inputs, targets=self.data[self.name]['summary'])
        self.gen_inputs_idx['decoder_input_ids'] = SingleExt.shift_tokens_right(self.gen_inputs_idx['labels'],
                                                         self.tokenizer.pad_token_id,
                                                         self.decoder_start_token_id)

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def decode(self, summary: str) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def gen_init_inputs(self, use_mask=False):

        self.gen_inputs = []

        for idx, line in tqdm(enumerate(self.data[self.name]['summary']), desc='gen_init...', leave=True):

            context = self.data[self.name]['text'][idx]
            if use_mask:
                mask = '<mask>'
                self.gen_inputs.append(f"{context} Head Entity : {mask} , Tail Entity : {mask} , Relation : {mask} .")
            else:
                self.gen_inputs.append(context)

    def get_vocab(self, path_in: str):

        raise NotImplementedError

    @classmethod
    def load(cls, path: str, name: str):
        data_files = {}
        data_files[name] = path
        extension = path.split(".")[-1]
        sents = load_dataset(
            extension, data_files=data_files, cache_dir=None
        )
        return sents

    def write_data(self, data: wr_Dataset, name: str) -> str:
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = ExtractEncoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(self.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))

        return str(path_out)

    def get_data(self, path_in: str, name: str):
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / f"{name}.json"
        if Path(str(path_out)).exists():
            dataset = self.load(str(path_out), name)
        else:
            path_in = Path(path_in) / f"{name}.jsonl"
            data = wr_Dataset.load(str(path_in))
            path_out = self.write_data(data, name)
            dataset = self.load(path_out, name)

        return dataset


class SingleExtTr(Dataset):
    def __init__(self, name, path, model_args, train_args, data_args, vae_args,
                 data_name='u2t_map_all', split='unseen_10_seed_0/'):
        self.path_in = "outputs/data/splits/zero_rte/" + data_name + "/" + split
        self.path = path
        self.name = name
        self.save_dir = str(Path(path) / "extractor")
        self.random_seed = train_args.seed
        self.batch_sz = train_args.per_device_train_batch_size
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.padding = "max_length" if data_args.pad_to_max_length else True #False
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss
        self.decoder_start_token_id = 2  # same as BartConfig
        if torch.cuda.is_available() and len(vae_args.trainer_params.gpus) != 0:
            self.cuda = True
        else:
            self.cuda = False

        tokenizer_kwargs = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
                        model_args.tokenizer_name
                        if model_args.tokenizer_name
                        else model_args.model_name_or_path,
                        cache_dir=model_args.cache_dir,
                        use_fast=model_args.use_fast_tokenizer,
                        revision=model_args.model_revision,
                        use_auth_token=True if model_args.use_auth_token else None,
                        **tokenizer_kwargs, )


    def __getitem__(self, index):


        data = [self.gen_inputs_idx['input_ids'][index],
               self.gen_inputs_idx['attention_mask'][index],
               self.gen_inputs_idx['labels'][index],
               self.gen_inputs_idx['decoder_input_ids'][index]]

        item = self.collate_fn(data)

        return item

    def __len__(self):
        return self.len

    def collate_fn(self, data):

        dat = pd.DataFrame(data)[0]

        ext_inp = pad_sequence(dat[0], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ext_att = pad_sequence(dat[1], batch_first=True, padding_value=0)
        data_ext = {"input_ids": ext_inp, "attention_mask": ext_att,
                    "labels": pad_sequence(dat[2], batch_first=True, padding_value=-100),
                    "decoder_input_ids": pad_sequence(dat[3], batch_first=True, padding_value=self.tokenizer.pad_token_id)}

        return data_ext

    def initialize(self):
        if self.name == 'synthetic':
            self.data = self.get_data(self.path, self.name)
        else:
            self.data = self.get_data(self.path_in, self.name)
        self.gen_init_inputs()
        self.preprocess_fun()

        self.len = len(self.data[self.name]['text'])
        print('\n dataloader initialized!')

    def compute_dis(self, mus, logvars):
        raise NotImplementedError

    def vae_sampling(self, model:Any, k=1):
        raise NotImplementedError

    def tokenize(self, inputs, tokenizer=None, targets=None):

        if tokenizer is not None:
            inps, rel_ids = inputs
            model_inputs = {"input_ids":[], "relation_ids":rel_ids}
            for inp in inps:
                model_inputs['input_ids'].append(tokenizer.to_indices(inp))
        else:
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_source_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt"
            )
            if targets is not None:
                # Setup the tokenizer for targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        targets, max_length=self.max_target_length, padding=self.padding, truncation=True,
                        return_tensors="pt"
                    )

                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                        for label in labels["input_ids"]
                    ]

                model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_fun(self):

        self.gen_inputs_idx = self.tokenize(self.gen_inputs, targets=self.data[self.name]['summary'])
        self.gen_inputs_idx['decoder_input_ids'] = SingleExt.shift_tokens_right(self.gen_inputs_idx['labels'],
                                                         self.tokenizer.pad_token_id,
                                                         self.decoder_start_token_id)

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def decode(self, summary: str) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def gen_init_inputs(self, use_mask=False):

        self.gen_inputs = []

        for idx, line in tqdm(enumerate(self.data[self.name]['summary']), desc='gen_init...', leave=True):

            context = self.data[self.name]['text'][idx]
            if use_mask:
                mask = '<mask>'
                self.gen_inputs.append(f"{context} Head Entity : {mask} , Tail Entity : {mask} , Relation : {mask} .")
            else:
                self.gen_inputs.append(context)

    def get_vocab(self, path_in: str):

        raise NotImplementedError

    @classmethod
    def load(cls, path: str, name: str):
        data_files = {}
        data_files[name] = path
        extension = path.split(".")[-1]
        sents = load_dataset(
            extension, data_files=data_files, cache_dir=None
        )
        return sents

    def write_data(self, data: wr_Dataset, name: str) -> str:
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = ExtractEncoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(self.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))

        return str(path_out)

    def get_data(self, path_in: str, name: str):
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / f"{name}.json"
        if Path(str(path_out)).exists():
            dataset = self.load(str(path_out), name)
        else:
            path_in = Path(path_in) / f"{name}.jsonl"
            data = wr_Dataset.load(str(path_in))
            path_out = self.write_data(data, name)
            dataset = self.load(path_out, name)

        return dataset


class ExtDataTr(SingleExtTr):
    def __init__(self, name, path, model_args, train_args, data_args, vae_args,
                 data_name='u2t_map_all', split='unseen_10_seed_0/', threshold_0=0.3, threshold_1=0.6):
        super().__init__(name, path, model_args, train_args, data_args, vae_args, split=split)
        path_in = "outputs/data/splits/zero_rte/" + data_name + "/" + split
        self.path = path
        self.get_vocab(path_in)
        self.threshold_0 = threshold_0
        self.threshold_1 = threshold_1
        if name == 'synthetic':
            self.data = self.get_data(path, name)
            # self.data_train = self.get_data(path_in, 'train')
        else:
            self.data = self.get_data(path_in, name)

        self.gen_init_inputs()

        self.len = len(self.data[name]['text'])

        print('\n dataloader initialized!')

    def __getitem__(self, index):

         return self.gen_inputs_idx['input_ids'][index], \
               self.gen_inputs_idx['attention_mask'][index], \
               self.gen_inputs_idx['labels'][index], \
               self.gen_inputs_idx['decoder_input_ids'][index], \
               self.ctr_inputs_idx_pos['input_ids'][index], \
               self.ctr_inputs_idx_pos['attention_mask'][index], \
               self.ctr_inputs_idx_neg[index]['input_ids'], \
               self.ctr_inputs_idx_neg[index]['attention_mask']

    def __len__(self):
        return self.len

    def collate_fn(self, data):

        dat = pd.DataFrame(data)
        ext_inp = []

        ext_inp.extend(dat[4])
        for item in dat[6]:
            ext_inp.extend(item)

        ext_inp = pad_sequence(ext_inp, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        ext_att = []
        ext_att.extend(dat[5])
        for item in dat[7]:
            ext_att.extend(item)

        ext_att = pad_sequence(ext_att, batch_first=True, padding_value=0)
        data_ext = {"input_ids": pad_sequence(dat[0], batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    "attention_mask": pad_sequence(dat[1], batch_first=True, padding_value=0),
                    "labels": pad_sequence(dat[2], batch_first=True, padding_value=-100),
                    "decoder_input_ids": pad_sequence(dat[3], batch_first=True, padding_value=self.tokenizer.pad_token_id)}

        data_cnt = {"input_ids": ext_inp, "attention_mask": ext_att}

        return [data_ext, data_cnt]

    def compute_dis(self, mus, logvars):
        """
        KL((m0,s0)||(m1,s1))
        =.5 * (log(|s1|/|s0|) + tr(s1^(-1)s0) + (m1-m0)^T s1^{-1} (m1-m0) - N)
        This function deals with KL-Divergence between any two multivariate normal distribution
        in batch.
        :param mus:
        :param logvars:
        :return:
        """
        n = logvars.size(0)
        logvars_ = torch.diag_embed(logvars)
        logdet_var = torch.sum(logvars, dim=1) # because of log operation on vars [B]
        logdet_var1 = logdet_var.unsqueeze(1).repeat(1, n)
        logdet_var2 = logdet_var.unsqueeze(0).repeat(n, 1)
        item1 = logdet_var1 - logdet_var2  # [B, B]

        std_2 = torch.exp(logvars_)
        inv_std_2 = torch.inverse(std_2)  # [B, d, d] it's a diagonal matrix
        inv_std_2_vec = torch.diagonal(inv_std_2, dim1=1, dim2=2) # [B, d]
        item2 = torch.mm(inv_std_2_vec, torch.diagonal(std_2, dim1=1, dim2=2).transpose(0, 1))  # [B, B]

        mu1 = mus.unsqueeze(1).repeat(1, n, 1)  # [B, B, d]
        mu2 = mus.unsqueeze(0).repeat(n, 1, 1)  # [B, B, d]
        mu2_1_sq = (mu2-mu1) * (mu2-mu1)  # [B, B, d]
        item3 = torch.matmul(mu2_1_sq, inv_std_2_vec.unsqueeze(2)).squeeze(-1)  # [B, B]

        kl = 0.5 * (item1 + item2 + item3 - n)

        return kl

    def vae_sampling(self, model: Any, k=3):
        self.vae_keys = list(self.vae_dic.keys())
        vae_sents = []
        vae_relations = []
        for key in self.vae_keys:
            vae_sents.append(random.choice(self.vae_dic[key])['context'])
            # vae_relations.append(self.vae_keys.index(key))
            vae_relations.append(self.rel_vocab[key])
        vae_relations = torch.LongTensor(vae_relations)

        vae_dataloader, _ = get_dataloader([[vae_sents, vae_relations], self.vae_vocab], model_type='vae', shuffle=False)

        # get topk most close relations
        model.eval()
        mus = []
        logvars = []
        with torch.no_grad():
            for data in vae_dataloader:
                if self.cuda:
                    data = [dt.cuda() for dt in data]

                _, _, mu, logvar = model(data[0], relations=data[2], input_length=data[1].squeeze(1))
                mus.append(mu)
                logvars.append(logvar)
            mus = torch.cat(mus, dim=0)
            logvars = torch.cat(logvars, dim=0)

        kl_div = -1.*self.compute_dis(mus, logvars)
        kl_div.fill_diagonal_(float('-inf'))
        # for each p2 (Q appoximate distribution) select topk most close k p1 (P true distribution)
        indices = torch.topk(kl_div, k=k, dim=1, largest=True)[1]  # [B, k] B is the total number of relations

        # indices -> int selected vae_relations -> str relations
        self.selected_relations_idx = vae_relations[indices]

    def preprocess_fun(self):

        self.ctr_inputs_neg = []
        self.vae_rels = []

        for idx, line in enumerate(self.data[self.name]['text']):

            ctr_inps = []
            true_rel = self.relations[idx]
            relative_idx = self.vae_keys.index(true_rel)
            true_rel_idx = self.rel_vocab[true_rel]
            self.vae_rels.append(true_rel_idx)

            for rel_idx in self.selected_relations_idx[relative_idx]:

                neg_sample = random.choice(self.vae_dic[self.rev_rel_vocab[rel_idx.item()]])
                neg_cnt = neg_sample["context"]
                ctr_inps.append("Context : " + neg_cnt + " " + self.data[self.name]['summary'][idx])

            self.ctr_inputs_neg.append(ctr_inps)

        self.gen_inputs_idx = self.tokenize(self.gen_inputs, targets=self.data[self.name]['summary'])
        self.gen_inputs_idx['decoder_input_ids'] = SingleExt.shift_tokens_right(self.gen_inputs_idx['labels'],
                                                         self.tokenizer.pad_token_id,
                                                         self.decoder_start_token_id)
        self.ctr_inputs_idx_pos = self.tokenize(self.ctr_inputs_pos)
        self.ctr_inputs_idx_neg = [self.tokenize(item) for item in self.ctr_inputs_neg]

    def preprocess_fun_(self):

        self.ctr_inputs_neg = []
        self.vae_rels = []

        for idx, line in enumerate(self.data[self.name]['text']):

            ctr_inps = []
            true_rel = self.relations[idx]
            relative_idx = self.vae_keys.index(true_rel)
            true_rel_idx = self.rel_vocab[true_rel]
            self.vae_rels.append(true_rel_idx)

            for rel_idx in self.selected_relations_idx[relative_idx]:

                neg_sample = random.choice(self.vae_dic[self.rev_rel_vocab[rel_idx.item()]])
                neg_cnt = neg_sample["context"]
                neg_id = neg_sample["index"]
                neg_head = self.heads[neg_id]
                neg_tail = self.tails[neg_id]
                head, tail, relation = self.decode(self.data[self.name]['summary'][idx])
                if random.random() < self.threshold_0:
                    ctr_inps.append(
                        f"Context : {neg_cnt} Head Entity : {neg_head} , Tail Entity : {relation} , Relation : {tail} .")
                elif random.random() >= self.threshold_0 and random.random() < self.threshold_1:
                    ctr_inps.append(
                        f"Context : {neg_cnt} Head Entity : {head} , Tail Entity : {relation} , Relation : {neg_tail} .")
                else:
                    ctr_inps.append(
                        f"Context : {neg_cnt} Head Entity : {head} , Tail Entity : {relation} , Relation : {tail} .")

            self.ctr_inputs_neg.append(ctr_inps)

        self.gen_inputs_idx = self.tokenize(self.gen_inputs, targets=self.data[self.name]['summary'])
        self.gen_inputs_idx['decoder_input_ids'] = SingleExt.shift_tokens_right(self.gen_inputs_idx['labels'],
                                                         self.tokenizer.pad_token_id,
                                                         self.decoder_start_token_id)
        self.ctr_inputs_idx_pos = self.tokenize(self.ctr_inputs_pos)
        self.ctr_inputs_idx_neg = [self.tokenize(item) for item in self.ctr_inputs_neg]

    def decode(self, summary: str) -> Tuple[Any, Any, Any]:
        front, back = summary.split(" , Relation : ")
        relation = back.split(' .')[0]
        front_, tail = front.split(" , Tail Entity :")
        head = front_.split("Head Entity : ")[-1]
        return head, tail, relation

    def gen_init_inputs(self, use_mask=False):
        self.gen_inputs = []
        self.ctr_inputs_pos = []
        self.heads = []
        self.tails = []
        self.relations = []
        self.vae_dic = {}
        for idx, line in tqdm(enumerate(self.data[self.name]['summary']), desc='gen_init...', leave=True):
            head, tail, relation = self.decode(line)
            self.heads.append(head)
            self.tails.append(tail)
            self.relations.append(relation)

            context = self.data[self.name]['text'][idx]
            if use_mask:
                mask = '<mask>'
                self.gen_inputs.append(f"{context} Head Entity : {mask} , Tail Entity : {mask} , Relation : {mask} .")
            else:
                self.gen_inputs.append(context)

            self.ctr_inputs_pos.append(context + " " + line) 
            vae_inp = context.split("Context : ")[-1]
            if relation not in self.vae_dic:
                self.vae_dic[relation] = [{'context': vae_inp, 'index': idx}]
            else:
                self.vae_dic[relation].append({'context': vae_inp, 'index': idx})
            # self.vae_vocab.add_sentence(vae_inp)

    def gen_init_inputs_(self, use_mask=False):
        self.gen_inputs = []
        self.ctr_inputs_pos = []
        self.heads = []
        self.tails = []
        self.relations = []
        self.vae_dic = {}
        sample_id = 0
        # for name in ['train', self.name]:
        datas = [('train', self.data_train['train']), (self.name, self.data[self.name])] if self.name == 'synthetic' else [(self.name, self.data[self.name])]
        for (data_name, dat) in datas:
            for idx, line in tqdm(enumerate(dat['summary']), desc='gen_init...', leave=True):
                head, tail, relation = self.decode(line)
                self.heads.append(head)
                self.tails.append(tail)
                self.relations.append(relation)

                context = dat['text'][idx]
                if data_name != 'train':
                    if use_mask:
                        mask = '<mask>'
                        self.gen_inputs.append(f"{context} Head Entity : {mask} , Tail Entity : {mask} , Relation : {mask} .")
                    else:
                        self.gen_inputs.append(context)

                    self.ctr_inputs_pos.append(context + " " + line) 

                vae_inp = context.split("Context : ")[-1]
                if relation not in self.vae_dic:
                    self.vae_dic[relation] = [{'context': vae_inp, 'index': sample_id}]
                else:
                    self.vae_dic[relation].append({'context': vae_inp, 'index': sample_id})

                sample_id += 1

    def get_vocab(self, path_in: str):
        self.vae_vocab = Vocabulary()
        self.rev_rel_vocab = {}
        data_dir = str(Path(self.save_dir) / "data")
        path_out = Path(data_dir) / "vae_vocab.json"
        path_out_rel = Path(data_dir) / "rel_vocab.json"
        if Path(str(path_out)).exists():
            with open(path_out, 'r') as f:
                self.vae_vocab.word2index = json.load(f)
            self.vae_vocab.update_vocab()

            with open(path_out_rel, 'r') as f:
                self.rel_vocab = json.load(f)

        else:
            self.rel_vocab = {}
            for name in ['train', 'dev', 'synthetic']:
                if name == 'synthetic':
                    dataset = self.get_data(self.path, name)
                else:
                    dataset = self.get_data(path_in, name)
                for idx, line in tqdm(enumerate(dataset[name]['text']), desc='get_vae_vocab', leave=True):
                    head, tail, relation = self.decode(dataset[name]['summary'][idx])
                    if relation not in self.rel_vocab:
                        self.rel_vocab[relation] = len(self.rel_vocab)
                    vae_inp = line.split("Context : ")[-1]
                    self.vae_vocab.add_sentence(vae_inp)
            with open(path_out, 'w') as f:
                json.dump(self.vae_vocab.word2index, f)
            with open(path_out_rel, 'w') as f:
                json.dump(self.rel_vocab, f)

        # generate reverse_rel_vocab: index to relation
        for key in self.rel_vocab.keys():
            self.rev_rel_vocab[self.rel_vocab[key]] = key

        print("number of relations : ", len(self.rel_vocab))
        print("number of vae vocabularies : ", len(self.vae_vocab.word2index))


def get_dataloader(inps, model_type='vae', shuffle=True, bz=32, num_workers=1, pin_memory=False):

    tokenizer = None
    if model_type == 'vae':
        [vae_sents, vae_relations], vae_vocab = inps
        data = VAEData([vae_sents, vae_relations], vae_vocab)

    elif model_type in ['extraction', 'single_vae']:
        data = inps 

    else:

        name, path, model_args, train_args, data_args, vae_args, data_name, split = inps
        data = SingleExt(name, path, model_args, train_args, data_args, vae_args,
                         data_name=data_name, split=split)
        data.initialize()
        tokenizer = data.tokenizer

    dataloader = DataLoader(data,
                            batch_size=bz,
                            shuffle=shuffle,
                            collate_fn=data.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory
                            )

    return dataloader, tokenizer

