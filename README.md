# ParallelTokenizers

Abstract—Tokenization is an influential part of the Transformer model. Breaking down the text into the units to analyze significantly affects the quality of text processing. There exist multiple algorithms producing different tokenizations. All the tokenization algorithms have their advantages and disadvantages, so combining various tokenization algorithms in a single model may compensate for the disadvantages and be beneficial for the quality of text processing. In this project, we explore an experimental technique that combines three different tokenizers to investigate its potential impact on model performance.

### Files

- **`updated_architechture.ipynb`**  
  A Jupyter notebook that outlines the updated architecture for the tokenization process and training set up.

- **`combined_tokenization_approaches.ipynb`**  
  A Jupyter notebook that explores and compares different tokenization approaches.

- **`summarization_inference.py`**  
  A Python script for running inference using the updated, trained model and trained tokenizers.

- **`trained_tokenizers/`**  
  A directory containing pre-trained tokenizers:
  - **`unigram/`**  
    Includes files for the unigram tokenizer:
    - `special_tokens_map.json`
    - `tokenizer_config.json`
    - `tokenizer.json`: 
  - **`wordpiece/`**  
    Includes files for the wordpiece tokenizer:
    - `special_tokens_map.json`
    - `tokenizer_config.json`
    - `tokenizer.json`
