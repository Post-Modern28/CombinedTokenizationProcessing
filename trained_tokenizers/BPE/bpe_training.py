from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("vblagoje/cc_news")

texts = [f"{title} {desc} {text}" for title, desc, text in zip(ds["train"]["title"], ds["train"]["description"], ds["train"]["text"])]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=128256,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
)

print("Training tokenizer...")
tokenizer.train_from_iterator(tqdm(texts, desc="Training Progress"), trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))]
)

tokenizer.save("bpe_tokenizer.json")