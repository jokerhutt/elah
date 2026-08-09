import numpy as np

from config import PRETRAIN_TOKENS, SFT_TOKENS, TOKEN_DTYPE, TOKENS_DIR
from dataset import format_chat, iter_pretraining_data, iter_sft_data

from .tokenizer import Tokenizer


def pretokenize():
    pretokenize_pretrain()
    pretokenize_sft()


def pretokenize_pretrain():
    _write_tokens(PRETRAIN_TOKENS, iter_pretraining_data())


def pretokenize_sft():
    _write_tokens(SFT_TOKENS, (format_chat(messages) for messages in iter_sft_data()))


def _write_tokens(path, texts):
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer()

    with open(path, "wb") as f:
        for text in texts:
            token_ids = tokenizer.encode(text) + [tokenizer.eot_id]

            np.array(token_ids, dtype=TOKEN_DTYPE).tofile(f)


if __name__ == "__main__":
    pretokenize()
