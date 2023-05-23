import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any
from copy import deepcopy
import torch
from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm
from nltk import word_tokenize, pos_tag

from generation import LabelConstraint, TripletSearchDecoder
from modeling import (NewRelationExtractor, RelationGenerator, RelationModel,
                      select_model)
from utils import (RelationSentence, delete_checkpoints, safe_divide)
from knowledge_prepare import load_u2t


class Sentence(BaseModel):
    triplets: List[RelationSentence]

    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0


class Dataset(BaseModel):
    sents: List[Sentence]

    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))

    def get_prompt(self) -> Dict[str, Any]:
        with open("./outputs/data/concept/entity2_to_cpt.json", "r") as f:
            entity2_to_cpt = json.load(f)

        labels = {}
        for s in self.sents:
            for t in s.triplets:
                entity2 = t.as_tuple()[2]
                rel = t.label
                if entity2 in entity2_to_cpt:
                    if rel not in labels:

                        labels[rel] = deepcopy(entity2_to_cpt[entity2])
                    else:

                        for cpt in entity2_to_cpt[entity2]:
                            if cpt not in labels[rel]:
                                labels[rel].append(cpt)

        return labels

    def get_symprompt(self) -> Dict[str, Any]:
        with open("./outputs/data/concept/rel_to_entities_.json", "r") as f:
            rel2ent = json.load(f)
        relations = self.get_labels()
        labels = {}
        # for rel in rel2ent.keys():
        for rel in relations:
            if rel == "have_no":
                rel2ent[rel] = rel2ent["own"]
            if rel in rel2ent:
                labels[rel] = []
                for value in rel2ent[rel]:
                    labels[rel].append([" ".join(value.split('_'))])
            else:
                print("cannot find label info: ", rel)

        return labels
        # return sorted(set(t.label + " Entity2 : " + t.as_tuple()[2] for s in self.sents for t in s.triplets))

    @classmethod
    def combine_label(cls, dev_l, test_l):
        for key in dev_l.keys():
            if key not in test_l:
                test_l[key] = deepcopy(dev_l[key])
            else:
                for cpt in dev_l[key]:
                    if cpt not in test_l[key]:
                        test_l[key].append(cpt)
        return test_l

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [Sentence(**json.loads(line)) for line in f]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")


    @classmethod
    def load_persona(cls):
        data = load_u2t(map_type='all')
        return cls(sents=data)

    def filter_labels(self, labels: List[str]):
        label_set = set(labels)
        sents = []
        for s in self.sents:
            triplets = [t for t in s.triplets if t.label in label_set]
            if triplets:
                s = s.copy(deep=True)
                s.triplets = triplets
                sents.append(s)
        return Dataset(sents=sents)

    def train_test_split(self, test_size: int, random_seed: int, by_mean: str):
        random.seed(random_seed)

        if by_mean == 'by_label':
            labels = self.get_labels()
            labels_test = random.sample(labels, k=test_size)
            labels_train = sorted(set(labels) - set(labels_test))
            sents_train = self.filter_labels(labels_train).sents
            sents_test = self.filter_labels(labels_test).sents
        elif by_mean == 'test_other':
            labels = self.get_labels().pop('other')
            labels_test = random.sample(labels, k=test_size)
            labels_train = sorted(set(labels) - set(labels_test))
            labels_test = labels_test + {'other'}
            sents_train = self.filter_labels(labels_train).sents
            sents_test = self.filter_labels(labels_test).sents
        else:
            sents_train = [s for s in self.sents]
            sents_test = random.sample(self.sents, k=test_size)

        banned = set(s.text for s in sents_test)  # Prevent sentence overlap
        sents_train = [s for s in sents_train if s.text not in banned]
        assert len(self.sents) == len(sents_train) + len(sents_test)
        return Dataset(sents=sents_train), Dataset(sents=sents_test)

    def analyze(self):
        info = dict(
            sents=len(self.sents),
            unique_texts=len(set(s.triplets[0].text for s in self.sents)),
            lengths=str(Counter(len(s.triplets) for s in self.sents)),
            labels=len(self.get_labels()),
        )
        print(json.dumps(info, indent=2))


def get_entity2_vocab(dev, test):
    sents_dev = dev.sents
    sents_test = test.sents
    pos_set = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
               "JJ", "JJR", "JJS", "NN", "NNS", "RB", "RBR", "RBS", "RP"]
    entity2_dic = {}
    vocab = {}
    for sents in [sents_dev, sents_test]:
        for sent in sents:
            for trp in sent.triplets:
                s, r, o = trp.as_tuple()
                tokens = word_tokenize(o)
                pos = pos_tag(tokens)
                phrase = []
                phrase_pos = set()
                entity2s = []
                for t in pos:
                    if t[1] in pos_set:
                        if len(phrase_pos) < 2:

                            phrase.append(t[0])
                            phrase_pos.add(t[1])
                        else:

                            entity2s.append(phrase)

                            phrase = [t[0]]
                            phrase_pos = set()
                            phrase_pos.add(t[1])

                entity2s.append(phrase)
                for p in entity2s:
                    if o in entity2_dic:
                        if " ".join(p) not in entity2_dic[o]:
                            entity2_dic[o].append(" ".join(p))
                        if " ".join(p) not in vocab:
                            vocab[" ".join(p)] = ""

                    else:
                        entity2_dic[o] = [" ".join(p)]
                        vocab[" ".join(p)] = ""
                    for t in p:
                        if t not in entity2_dic[o]:
                            entity2_dic[o].append(t)
                        if t not in vocab:
                            vocab[t] = ""

    with open("./outputs/data/concept/entity2_dic.json", "w") as f:
        json.dump(entity2_dic, f)

    with open("./outputs/data/concept/vocab.json", "w") as f:
        json.dump(vocab, f)

    return


def write_data_splits(
    path_in: str,
    mode: str,
    folder_out: str = "outputs/data/splits/zero_rte",
    num_dev_labels: int = 5,
    num_test_labels: List[int] = [5, 10, 15],
    seeds: List[int] = [0, 1, 2, 3, 4],
    by_mean: str = 'test_other'
):
    for n in num_test_labels:
        for s in seeds:
            if mode == "fewrel":
                data = Dataset.load_fewrel(path_in)
            elif mode == "wiki":
                data = Dataset.load_wiki(path_in)
            elif mode == "persona":
                data = Dataset.load_persona()
            else:
                raise ValueError()

            train, test = data.train_test_split(
                test_size=n, random_seed=s, by_mean=by_mean
            )
            train, dev = train.train_test_split(
                test_size=num_dev_labels, random_seed=s, by_mean=by_mean
            )
            del data

            if by_mean == 'test_other':
                folder_out = folder_out + "_other"

            for key, data in dict(train=train, dev=dev, test=test).items():
                name = f"unseen_{n}_seed_{s}"
                path = Path(folder_out) / Path(path_in).stem / name / f"{key}.jsonl"
                data.save(str(path))
                print(dict(key=key, labels=len(data.get_labels()), path=path))


class Generator(BaseModel):
    load_dir: str
    save_dir: str
    num_gen_per_label: int = 250
    model_name: str = "generate"
    encoder_name: str = "generate"
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: Dataset, name: str, tail: bool=True) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.txt"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = model.get_encoder()
        lines = [encoder.encode_to_line(t, tail=tail) for s in data.sents for t in s.triplets]
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str, tail: bool=True):
        model = self.get_model()
        if Path(model.model_dir).exists():
            print("model directory already exists:", model.model_dir)
            return

        data_train = Dataset.load(path_train)
        data_dev = Dataset.load(path_dev)
        path_train = self.write_data(data_train, "train", tail=tail)
        path_dev = self.write_data(data_dev, "dev", tail=tail)
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def generate_(self, labels: Dict[str, Any], path_out: str):
        if Path(path_out).exists():
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for relation in tqdm(labels):
            triplets, raw = model.generate_(relation, labels[relation],
                                            self.num_gen_per_label, pipe=pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def generate(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for relation in tqdm(labels):
            triplets, raw = model.generate(relation, self.num_gen_per_label, pipe=pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)


class Extractor(BaseModel):
    load_dir: str
    save_dir: str
    model_name: str = "new_extract"
    encoder_name: str = "extract"
    search_threshold: float = -0.9906
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: Dataset, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = model.get_encoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str):
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        data_train = Dataset.load(path_train)
        data_dev = Dataset.load(path_dev)
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def predict(self, path_in: str, path_out: str, use_label_constraint: bool = True):
        data = Dataset.load(path_in)
        texts = [s.text for s in data.sents]
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
        gen = model.load_generator(torch.device("cuda"))
        encoder = model.get_encoder()
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        sents = []

        for i in tqdm(range(0, len(texts), model.batch_size)):
            batch = texts[i : i + model.batch_size]
            x = [encoder.encode_x(t) for t in batch]
            outputs = model.gen_texts(
                x, gen, num_beams=1, save_scores=use_label_constraint
            )
            assert len(outputs) == len(x)

            for j, raw in enumerate(outputs):
                triplet = encoder.safe_decode(x[j], y=raw)
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[j])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def predict_multi(self, path_in: str, path_out: str):
        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")
        print(dict(predict_multi=locals()))
        data = Dataset.load(path_in)
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
        gen = model.load_generator(torch.device("cuda"))
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        searcher = TripletSearchDecoder(
            gen=gen, encoder=model.get_encoder(), constraint=constraint
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text))
            for s in tqdm(data.sents)
        ]
        Dataset(sents=sents).save(path_raw)
        for s in sents:
            s.triplets = [t for t in s.triplets if t.score > self.search_threshold]
        Dataset(sents=sents).save(path_out)

    @staticmethod
    def score(path_pred: str, path_gold: str) -> dict:
        pred = Dataset.load(path_pred)
        gold = Dataset.load(path_gold)
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


def main(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    tail: bool=True
):
    print(dict(main=locals()))
    # generator = Generator(
    #     load_dir="gpt2",
    #     save_dir=str(Path(save_dir) / "generator"),
    # )
    # extractor = Extractor(
    #     load_dir="facebook/bart-base",
    #     save_dir=str(Path(save_dir) / "extractor"),
    # )

    # generator.fit(path_train, path_dev)
    # extractor.fit(path_train, path_dev)
    # path_synthetic = str(Path(save_dir) / "synthetic.jsonl")

    # if tail:
    #     labels_dev = Dataset.load(path_dev).get_prompt()
    #     labels_test = Dataset.load(path_test).get_prompt()
    #     labels = Dataset.combine_label(labels_dev, labels_test)
    #     generator.generate_(labels, path_out=path_synthetic)
    # else:
    #     labels_dev = Dataset.load(path_dev).get_labels()
    #     labels_test = Dataset.load(path_test).get_labels()
    #     generator.generate(labels_dev + labels_test, path_out=path_synthetic)

    # extractor_final = Extractor(
    #     load_dir=str(Path(save_dir) / "extractor" / "model"),
    #     save_dir=str(Path(save_dir) / "extractor_final"),
    # )
    # extractor_final.fit(path_synthetic, path_dev)

    extractor_final = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor"),
    )

    path_pred = str(Path(save_dir) / "pred.jsonl")
    extractor_final.predict(path_in=path_test, path_out=path_pred)
    results = extractor_final.score(path_pred, path_test)
    print(json.dumps(results, indent=2))
    with open(Path(save_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    # write_data_splits(path_in='./data/ConvAI2/u2t_map_all.json', mode='persona')
    # dev = Dataset.load("outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/dev.jsonl")
    #
    # test = Dataset.load("outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/test.jsonl")
    # get_entity2_vocab(dev, test)
    main(path_train="outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/train.jsonl",
         path_dev="outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/dev.jsonl",
         path_test="outputs/data/splits/zero_rte/u2t_map_all/unseen_10_seed_0/test.jsonl",
         save_dir="outputs/wrapper/u2t_map_all/unseen_10_seed_0")
    # Fire()

