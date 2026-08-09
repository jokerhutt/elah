import sentencepiece as spm

from config import MODEL_TYPE, SPECIAL_TOKENS, TOKENIZER_DIR, TOKENIZER_PREFIX, VOCAB_SIZE
from dataset import iter_tokenizer_data


def train_tokenizer():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_tokenizer_data(),
        model_prefix=str(TOKENIZER_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,

        user_defined_symbols=SPECIAL_TOKENS,

        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )


if __name__ == "__main__":
    train_tokenizer()
