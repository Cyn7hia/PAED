import json
import tqdm
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel

import torch


def train_test_split(*args, **kwargs) -> list:
    raise NotImplementedError


class RelationSentence(BaseModel):
    tokens: List[str]
    head: List[int]
    tail: List[int]
    label: str
    head_id: str = ""
    tail_id: str = ""
    label_id: str = ""
    error: str = ""
    raw: str = ""
    score: float = 0.0
    zerorc_included: bool = True

    def as_tuple(self) -> Tuple[str, str, str]:
        head = " ".join([self.tokens[i] for i in self.head])
        tail = " ".join([self.tokens[i] for i in self.tail])
        return head, self.label, tail

    def as_line(self) -> str:
        return self.json() + "\n"

    def is_valid(self) -> bool:
        for x in [self.tokens, self.head, self.tail, self.label]:
            if len(x) == 0:
                return False
        for x in [self.head, self.tail]:
            if -1 in x:
                return False
        return True

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @classmethod
    def from_spans(cls, text: str, head: str, tail: str, label: str, strict=True):
        tokens = text.split()
        sent = cls(
            tokens=tokens,
            head=find_span(head, tokens),
            tail=find_span(tail, tokens),
            label=label,
        )
        if strict:
            assert sent.is_valid(), (head, label, tail, text)
        return sent

    def as_marked_text(self) -> str:
        tokens = list(self.tokens)
        for i, template in [
            (self.head[0], "[H {}"),
            (self.head[-1], "{} ]"),
            (self.tail[0], "[T {}"),
            (self.tail[-1], "{} ]"),
        ]:
            tokens[i] = template.format(tokens[i])
        return " ".join(tokens)


def find_span(span: str, tokens: List[str]) -> List[int]:
    if span == "":
        return []
    start = find_sublist_index(tokens, span.split())
    if start >= 0:
        return [start + i for i in range(len(span.split()))]
    else:
        res = find_index(tokens, span.split())

        if res != -1:
            return res
        else:
            start, end = align_span_to_tokens(span, tokens)
            return list(range(start, end))


def find_sublist_index(items: list, query: list):
    length = len(query)
    for i in range(len(items) - length + 1):
        if items[i : i + length] == query:
            return i
    return -1


def find_index(items: list, query: list):
    indices = []

    for token in query:

        if token in items:
            indices.append(items.index(token))
        else:
            return -1
    return indices


def align_span_to_tokens(span: str, tokens: List[str]) -> Tuple[int, int]:
    # Eg align("John R. Allen, Jr.", ['John', 'R.', 'Allen', ',', 'Jr.'])
    char_word_map = {}
    num_chars = 0
    for i, w in enumerate(tokens):
        for _ in w:
            char_word_map[num_chars] = i
            num_chars += 1
    char_word_map[num_chars] = len(tokens)

    query = span.replace(" ", "")
    text = "".join(tokens)
    assert query in text
    i = text.find(query)
    start = char_word_map[i]
    end = char_word_map[i + len(query) - 1]
    assert 0 <= start <= end
    return start, end + 1


def delete_checkpoints(
    folder: str = ".", pattern="**/checkpoint*", delete: bool = True
):
    for p in Path(folder).glob(pattern):
        if (p.parent / "config.json").exists():
            print(p)
            if delete:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    os.remove(p)
                else:
                    raise ValueError("Unknown Type")


class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class RelationData(BaseModel):
    sents: List[RelationSentence]

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            lines = f.readlines()
            sents = [
                RelationSentence(**json.loads(x))
                for x in tqdm(lines, desc="RelationData.load")
            ]
        return cls(sents=sents)

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            f.write("".join([s.as_line() for s in self.sents]))

    @property
    def unique_labels(self) -> List[str]:
        return sorted(set([s.label for s in self.sents]))

    def train_test_split(
        self, test_size: Union[int, float], random_seed: int, by_label: bool = False
    ):
        if by_label:
            labels_train, labels_test = train_test_split(
                self.unique_labels, test_size=test_size, random_state=random_seed
            )
            train = [s for s in self.sents if s.label in labels_train]
            test = [s for s in self.sents if s.label in labels_test]
        else:
            groups = self.to_sentence_groups()
            keys_train, keys_test = train_test_split(
                sorted(groups.keys()), test_size=test_size, random_state=random_seed
            )
            train = [s for k in keys_train for s in groups[k]]
            test = [s for k in keys_test for s in groups[k]]

        # Enforce no sentence overlap
        texts_test = set([s.text for s in test])
        train = [s for s in train if s.text not in texts_test]

        data_train = RelationData(sents=train)
        data_test = RelationData(sents=test)
        if by_label:
            assert len(data_test.unique_labels) == test_size
            assert not set(data_train.unique_labels).intersection(
                data_test.unique_labels
            )

        info = dict(
            sents_train=len(data_train.sents),
            sents_test=len(data_test.sents),
            labels_train=len(data_train.unique_labels),
            labels_test=len(data_test.unique_labels),
        )
        print(json.dumps(info, indent=2))
        return data_train, data_test

    def to_sentence_groups(self) -> Dict[str, List[RelationSentence]]:
        groups = {}
        for s in self.sents:
            groups.setdefault(s.text, []).append(s)
        return groups

    def to_label_groups(self) -> Dict[str, List[RelationSentence]]:
        groups = {}
        for s in self.sents:
            groups.setdefault(s.label, []).append(s)
        return groups

    def filter_group_sizes(self, min_size: int = 0, max_size: int = 999):
        groups = self.to_sentence_groups()
        sents = [
            s
            for k, lst in groups.items()
            for s in lst
            if min_size <= len(lst) <= max_size
        ]
        return RelationData(sents=sents)

    def filter_errors(self):
        def check_valid_span(span: List[int]) -> bool:
            start = sorted(span)[0]
            end = sorted(span)[-1] + 1
            return span == list(range(start, end))

        sents = []
        for s in self.sents:
            if s.is_valid():
                if check_valid_span(s.head) and check_valid_span(s.tail):
                    sents.append(s)

        print(dict(filter_errors_success=len(sents) / len(self.sents)))
        return RelationData(sents=sents)

    def analyze(self, header: Optional[str] = None):
        labels = self.unique_labels
        groups = self.to_sentence_groups()
        spans = []
        words = []
        for s in self.sents:
            head, label, tail = s.as_tuple()
            spans.append(head)
            spans.append(tail)
            words.extend(s.tokens)
        info = dict(
            header=header,
            sents=len(self.sents),
            labels=str([len(labels), labels]),
            unique_texts=len(groups.keys()),
            unique_spans=len(set(spans)),
            unique_words=len(set(words)),
            group_sizes=str(Counter([len(lst) for lst in groups.values()])),
        )
        print(json.dumps(info, indent=2))
        return info


def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b


def _get_learning_rate(lr_scheduler):

    last_lr = lr_scheduler.get_last_lr()[0]
    if torch.is_tensor(last_lr):
        last_lr = last_lr.item()
    return last_lr


def load_u2t(map_type='diff'):

    with open('./data/ConvAI2/u2t_map_{}.json'.format(map_type), 'r') as f:
        u2t_map_load = json.load(f)

    return u2t_map_load
