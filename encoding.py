import re
from pathlib import Path
from typing import Dict, List, Tuple

from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk import word_tokenize

from transformer_base import run_summarization
from utils import RelationData, RelationSentence


class Encoder(BaseModel):
    def encode_x(self, x: str) -> str:
        raise NotImplementedError

    def encode_x_pro(self, x: str, t: str) -> str:
        raise NotImplementedError

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        raise NotImplementedError

    def decode(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    def decode_x(self, x: str) -> str:
        raise NotImplementedError

    def safe_decode(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x(x)
        try:
            s = self.decode(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        raise NotImplementedError

    def decode_from_line(self, line: str) -> RelationSentence:
        raise NotImplementedError

    def parse_line(self, line: str) -> Tuple[str, str]:
        raise NotImplementedError

    def encode_prompt(self, relation: str, ent: str)-> str:
        raise NotImplementedError

    def decode_(self, x: str, y: str) -> RelationSentence:
        raise NotImplementedError

    # fix no blanks between token and marks
    def re_line(self, line: str) -> str:
        # https://stackoverflow.com/questions/44263446/python-regex-to-add-space-after-dot-or-comma
        line_0 = re.sub(r'(?<=[.,:?"])(?=[^\s])', r' ', line)
        line = re.sub(r'(?=[.,:?"])(?<=[^\s])', r' ', line_0)
        return line


class GenerateEncoder(Encoder):
    def encode_x_(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Relation : {r} Entity2 : {o} ."

    def encode_x(self, r: str) -> str:
        return f"Relation : {r} ."

    def encode_prompt(self, r: str, tail: str):
        return f"Relation : {r} Entity2 : {tail} ."

    def decode_x(self, text: str) -> str:
        return text.split("Relation : ")[-1][:-2]

    def decode_prompt(self, text: str) -> Tuple[str, str]:
        r_raw, e_raw = text.split(" Entity2 : ")
        r = r_raw.split("Relation : ")[-1]
        e = e_raw[:-2]
        return r, e

    def encode_triplet(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Context : {sent.text} Head Entity : {s} , Tail Entity : {o} ."

    def decode_triplet(self, text: str, label: str) -> RelationSentence:
        front, back = text.split(" Head Entity : ")
        _, context = front.split("Context : ")
        head, back = back.split(" , Tail Entity : ")
        tail = back[:-2]
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_y_(self, sent: RelationSentence) -> str:
        return self.encode_x_(sent) + " " + self.encode_triplet(sent)

    def encode_y(self, sent: RelationSentence) -> str:
        return self.encode_x(sent.label) + " " + self.encode_triplet(sent)

    def decode_y_(self, text: str, label: str) -> RelationSentence:
        del label
        front, back = text.split(" . Context : ")
        label, tail = self.decode_prompt(front + " .")
        return self.decode_triplet("Context : " + back, label)  # Todo: tail==tail from decode_triplet?

    def decode_y(self, text: str, label: str) -> RelationSentence:
        del label
        front, back = text.split(" . Context : ")
        label = self.decode_x(front + " .")
        return self.decode_triplet("Context : " + back, label)

    def decode_(self, x: str, y: str) -> RelationSentence:
        r, tail = self.decode_prompt(x)
        sent = self.decode_y_(y, r)
        return sent

    def decode(self, x: str, y: str) -> RelationSentence:
        r = self.decode_x(x)
        sent = self.decode_y(y, r)
        return sent

    def encode_(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x_(sent)
        y = self.encode_y_(sent)
        return x, y

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.label)
        y = self.encode_y(sent)
        return x, y

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def encode_to_line(self, sent: RelationSentence, tail: bool=True) -> str:
        if tail:
            x, y = self.encode_(sent)
        else:
            x, y = self.encode(sent)
        return y + "\n"

    def parse_line(self, line: str) -> Tuple[str, str]:
        line_ = " ".join(word_tokenize(line.strip()))
        return "", line_
        # return "", line.strip()


class ExtractEncoder(Encoder):
    def encode_x(self, text: str) -> str:
        return f"Context : {text}"

    def encode_x_pro(self, text: str, t: str) -> str:
        if t == 'head':
            return f"Context : {text} Head Entity :"
        elif t == 'tail':
            return f"{text} Tail Entity :"
        elif t == 'relation':
            return f"{text} Relation :"

    def encode_x_(self, text: str) -> str:
        mask = '<mask>'
        return f"Context : {text} Head Entity : {mask} , Tail Entity : {mask} , Relation : {mask} ."

    def decode_x(self, x: str) -> str:
        return x.split("Context : ")[-1]

    def decode_x_(self, x: str) -> str:
        end = x.split("Context : ")[-1]
        return end.split(" Head Entity :")[0]

    def encode_y(self, sent: RelationSentence) -> str:
        s, r, o = sent.as_tuple()
        return f"Head Entity : {s} , Tail Entity : {o} , Relation : {r} ."

    def decode_y(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x(x)
        front, label = y.split(" , Relation : ")
        label = label[:-2]
        front, tail = front.split(" , Tail Entity : ")
        _, head = front.split("Head Entity : ")
        return RelationSentence.from_spans(context, head, tail, label)

    def decode_y_(self, x: str, y: str) -> RelationSentence:
        context = self.decode_x_(x)
        front, label = y.split(" , Relation : ")
        label = label[:-2]
        front, tail = front.split(" , Tail Entity : ")
        _, head = front.split("Head Entity : ")
        return RelationSentence.from_spans(context, head, tail, label)

    def encode_entity_prompt(self, head: str, tail: str) -> str:
        return f"Head Entity : {head} , Tail Entity : {tail} , Relation :"

    def encode(self, sent: RelationSentence) -> Tuple[str, str]:
        x = self.encode_x(sent.text)
        y = self.encode_y(sent)
        return x, y

    def decode(self, x: str, y: str) -> RelationSentence:
        return self.decode_y(x, y)

    def decode_(self, x: str, y: str) -> RelationSentence:
        return self.decode_y_(x, y)

    def safe_decode_(self, x: str, y: str) -> RelationSentence:
        text = self.decode_x_(x)
        try:
            s = self.decode_(x=x, y=y)
        except Exception as e:
            s = RelationSentence(
                tokens=text.split(), head=[], tail=[], label="", error=str(e), raw=y
            )
        return s

    def encode_to_line(self, sent: RelationSentence) -> str:
        x, y = self.encode(sent)
        return run_summarization.encode_to_line(x, y)

    def decode_from_line(self, line: str) -> RelationSentence:
        x, y = self.parse_line(line)
        return self.decode(x, y)

    def parse_line(self, line: str) -> Tuple[str, str]:
        return run_summarization.decode_from_line(line)


def test_encoders(
    paths: List[str] = [
        "outputs/data/zsl/wiki/unseen_5_seed_0/train.jsonl",
        "outputs/data/zsl/fewrel/unseen_5_seed_0/train.jsonl",
    ],
    print_limit: int = 4,
    encoder_names: List[str] = ["generate", "extract"],
    limit: int = 1000,
):
    encoders = {k: select_encoder(k) for k in encoder_names}

    for p in paths:
        data = RelationData.load(Path(p))
        _, data = data.train_test_split(min(limit, len(data.sents)), random_seed=0)

        for name, e in tqdm(list(encoders.items())):
            num_fail = 0
            print(dict(name=name, p=p))
            for s in data.sents:
                encoded = e.encode_to_line(s)
                x, y = e.parse_line(encoded)
                decoded: RelationSentence = e.safe_decode(x, y)

                if decoded.as_tuple() != s.as_tuple():
                    if num_fail < print_limit:
                        print(dict(gold=s.as_tuple(), text=s.text))
                        print(dict(pred=decoded.as_tuple(), text=decoded.text))
                        print(dict(x=x, y=y, e=decoded.error))
                        print()
                    num_fail += 1

            print(dict(success_rate=1 - (num_fail / len(data.sents))))
            print("#" * 80)


def select_encoder(name: str) -> Encoder:
    mapping: Dict[str, Encoder] = dict(
        extract=ExtractEncoder(),
        generate=GenerateEncoder(),
    )
    encoder = mapping[name]
    return encoder


def test_entity_prompts(
    path: str = "outputs/data/zsl/wiki/unseen_10_seed_0/test.jsonl", limit: int = 100
):
    def tokenize(text: str, tok) -> List[str]:
        return tok.convert_ids_to_tokens(tok(text, add_special_tokens=False).input_ids)

    data = RelationData.load(Path(path))
    e = ExtractEncoder()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    print(tokenizer)
    for i, s in enumerate(tqdm(data.sents[:limit])):
        head, label, tail = s.as_tuple()
        x, y = e.encode(s)
        prompt = e.encode_entity_prompt(head, tail)
        tokens_y = tokenize(y, tokenizer)
        tokens_prompt = tokenize(prompt, tokenizer)
        assert tokens_y[: len(tokens_prompt)] == tokens_prompt
        if i < 3:
            print(tokens_y)


if __name__ == "__main__":
    Fire()
