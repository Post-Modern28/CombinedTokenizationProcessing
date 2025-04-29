from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers import LlamaForCausalLM
import torch.nn as nn
import torch

login(token = "Your_HuggingFace_Token")


original_tokenizer = AutoTokenizer.from_pretrained("nickypro/tinyllama-110M")
original_tokenizer.padding_side = "right"


class LlamaWithExtraEmbeddings(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        original_vocab_size, embedding_dim = self.model.embed_tokens.weight.shape
        self.model.extra_embedding_1 = nn.Embedding(original_vocab_size, embedding_dim)
        self.model.extra_embedding_2 = nn.Embedding(original_vocab_size, embedding_dim)

model = LlamaWithExtraEmbeddings.from_pretrained("./saved_model")

from typing import List, Optional, Union
from cachetools import Cache
import types

def modified_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **kwargs
):
    if input_ids is None and inputs_embeds is None:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size = input_ids.shape[0]
    combined_embeds = []
    print(batch_size)
    try:
        for batch_idx in range(batch_size):
            str_input_ids = " ".join([str(i) for i in input_ids[batch_idx].tolist()])
            input_ids_parts = str_input_ids.split(" 200000 ")
            print(len(input_ids_parts), input_ids_parts)
            bpe = list(map(int, input_ids_parts[0].split(" ")))
            wordpiece = list(map(int, input_ids_parts[1].split(" ")))
            unigram = list(map(int, input_ids_parts[2].split(" ")))
            
            bpe += [original_tokenizer.pad_token_id] * (len(input_ids[batch_idx].tolist()) - len(bpe))
            wordpiece += [wordpiece_tokenizer.pad_token_id] * (len(input_ids[batch_idx].tolist()) - len(wordpiece))
            unigram += [unigram_tokenizer.pad_token_id] * (len(input_ids[batch_idx].tolist()) - len(unigram))

            bpe = torch.tensor(bpe, device=model.device).unsqueeze(0).long()
            wordpiece = torch.tensor(wordpiece, device=model.device).unsqueeze(0).long()
            unigram = torch.tensor(unigram, device=model.device).unsqueeze(0).long()
            
            bpe_embedding = self.model.embed_tokens(bpe)
            wordpiece_embedding = self.model.extra_embedding_1(wordpiece)
            unigram_embedding = self.model.extra_embedding_2(unigram)
            
            # print(f"Shapes: BPE {bpe_embedding.shape}, Unigram {unigram_embedding.shape}, SentencePiece {sentencepiece_embedding.shape}")
            
            min_length = min(bpe_embedding.shape[1], wordpiece_embedding.shape[1], unigram_embedding.shape[1])
            bpe_embedding = bpe_embedding[:, :min_length, :]
            wordpiece_embedding = wordpiece_embedding[:, :min_length, :]
            unigram_embedding = unigram_embedding[:, :min_length, :]
            
            batch_embeds = bpe_embedding + wordpiece_embedding + unigram_embedding
            combined_embeds.append(batch_embeds)

        inputs_embeds = torch.cat(combined_embeds, dim=0)
        # print(f"Final inputs_embeds shape: {inputs_embeds.shape}")
        
        if attention_mask is not None:
            attention_mask = attention_mask[batch_idx].tolist()
            attention_mask = attention_mask[:len(input_ids_parts[0].split(" "))]
            print(attention_mask)
            attention_mask += [0] * (len(input_ids[batch_idx].tolist()) - len(bpe))
            print(attention_mask)
            attention_mask = torch.tensor(attention_mask, device=model.device).unsqueeze(0).long()
        
        input_ids = None

    except IndexError as e:
        # print(input_ids, inputs_embeds)
        inputs_embeds = self.model.embed_tokens(input_ids)
        input_ids = None
        pass
        

    return self.original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        labels=labels,
        num_logits_to_keep=num_logits_to_keep,
        **kwargs
    )
    
model.original_forward = model.forward
model.forward = types.MethodType(modified_forward, model)


wordpiece_tokenizer = AutoTokenizer.from_pretrained("trained_tokenizers/wordpiece")
unigram_tokenizer = AutoTokenizer.from_pretrained("trained_tokenizers/unigram")
text = "Hello, how are you?"

original_tokenizer.pad_token_id = original_tokenizer.eos_token_id
unigram_tokenizer.pad_token_id = unigram_tokenizer.eos_token_id
wordpiece_tokenizer.pad_token_id = wordpiece_tokenizer.eos_token_id

original_tokens = original_tokenizer(text, padding='max_length', max_length=10)
wordpiece_tokens = wordpiece_tokenizer(text, padding='max_length', max_length=10)
unigram_tokens = unigram_tokenizer(text, padding='max_length', max_length=10)

combined_text = original_tokens["input_ids"] + [200000] + wordpiece_tokens["input_ids"] + [200000] + unigram_tokens["input_ids"]
combined_attention_mask = original_tokens['attention_mask'] + [200000] + wordpiece_tokens['attention_mask'] + [200000] + unigram_tokens['attention_mask']

output = model.generate(
    input_ids=torch.tensor([combined_text], device=model.device).long(),
    attention_mask=torch.tensor([combined_attention_mask], device=model.device).long(),
)

# print generated text
print("Generated text:")
for i in range(output.shape[0]):
    generated_text = original_tokenizer.decode(output[i], skip_special_tokens=True)
    print(generated_text)
