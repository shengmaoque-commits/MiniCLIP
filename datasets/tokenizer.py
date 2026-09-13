import re
import json
from collections import Counter


class SimpleTokenizer:
    def __init__(
        self,
        vocab=None,
        max_length=40
    ):
        self.max_length = max_length

        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
        }

        if vocab is None:
            self.vocab = dict(self.special_tokens)
        else:
            self.vocab = vocab
        #填充符号  （在规定的长度内，填充剩余的部分）
        self.pad_id = self.vocab["[PAD]"]
        #未知符号
        self.unk_id = self.vocab["[UNK]"]
        #分类符号 (可以看完整的句子 可以作为一整个的代表)
        self.cls_id = self.vocab["[CLS]"]
        #分隔符号 (可以看成是一个句子的结束符号)
        self.sep_id = self.vocab["[SEP]"]

    def tokenize(self, text):
        text = text.lower().strip()
        #正则  +表示重复多次 
        tokens = re.findall(
            r"[a-z0-9]+|[^\w\s]",
            text
        )

        return tokens

    def build_vocab(
        self,
        texts,
        min_freq=2,
        max_vocab_size=20000
    ):
        #计数用
        counter = Counter()

        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        words = [
            word
            for word, freq in counter.most_common()
            if freq >= min_freq
        ]

        words = words[:max_vocab_size - len(self.special_tokens)]
        #复制一份，而不是指向同一个
        self.vocab = dict(self.special_tokens)

        for word in words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

        print(
            f"Vocabulary size: {len(self.vocab)}"
        )

    def encode(self, text):
        tokens = self.tokenize(text)

        token_ids = [
            self.vocab.get(token, self.unk_id)
            for token in tokens
        ]

        # 留两个位置给 CLS 和 SEP
        token_ids = token_ids[:self.max_length - 2]

        token_ids = (
            [self.cls_id]
            + token_ids
            + [self.sep_id]
        )

        attention_mask = [1] * len(token_ids)

        padding_length = self.max_length - len(token_ids)

        token_ids += [self.pad_id] * padding_length
        attention_mask += [0] * padding_length

        return token_ids, attention_mask

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.vocab,
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, path, max_length=40):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return cls(
            vocab=vocab,
            max_length=max_length
        )

    def __len__(self):
        return len(self.vocab)