from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import Sequence, Lowercase, NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from datasets import load_dataset

dataset = load_dataset("vblagoje/cc_news")['train']
texts = dataset["text"]

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(vocab_size=128256, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.train_from_iterator(texts, trainer)

tokenizer_name = "wordpiece_tokenizer"
tokenizer.save(f"{tokenizer_name}.json")

print(f"Training is done. Tokenizer is saved as {tokenizer_name}.json")