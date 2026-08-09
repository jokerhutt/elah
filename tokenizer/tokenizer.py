from pathlib import Path

import sentencepiece as spm

from config import END_OF_TEXT, IM_END, IM_START, TOKENIZER_MODEL


class Tokenizer:
    def __init__(self, path: Path = TOKENIZER_MODEL):
        self.processor = spm.SentencePieceProcessor(model_file=str(path))

        self.eot_id = self.processor.piece_to_id(END_OF_TEXT)
        self.im_start_id = self.processor.piece_to_id(IM_START)
        self.im_end_id = self.processor.piece_to_id(IM_END)

    @property
    def vocab_size(self):
        return self.processor.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        # num_threads=-1 encodes the batch across all cores
        return self.processor.encode(texts, out_type=int, num_threads=-1)

    def decode(self, tokens: list[int]) -> str:
        return self.processor.decode(tokens)
