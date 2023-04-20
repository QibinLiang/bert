# BERT

This repository contains code for the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). The project is for me to learn about BERT and to experiment with it, so it might contain bugs and is not guaranteed to be correct. But it should be a good starting point for someone who wants to play with BERT. I have tried to train BERT on a small dataset and it seems to work. Although it can not generate a reasonable result, it somehow learns to predict the next word that is correct in terms of grammar. Of course, the model can fill the blank in a sentence where the token is `<mask>`.

## Quick Start

1. Download the wikitext-2 dataset. Note that you should manually move the dataset from the HuggingFace cache directory to the data directory.
   
```bash
python utils/download.py
```

2. Preprocess the dataset.
   
```bash
python utils/wikitext2_preprocess.py
```

3. Train the model.
   
```bash
python train.py
```
Parallel training is supported. You need to set `HYPER_PARAM.ddp` in `train.py` to True and run the following command.
   
```bash
sbatch train.sh
```

## Some Results
```
$ python inference.py

input a sentence and randomly mask some word by <mask>

sent:   getting out of <mask> because <mask> are not <mask> to get there .
pred:    getting out of view because they are not able to get there .

ent:   Cantonese is a <mask> of <mask> language that is <mask> the mother tongue of people <mask> the south of China . 
pred:    cantonese is a result of mitochondrial language that is probably the mother tongue of people encoding the south of china .

sent:   You <mask> do nothing more <mask> learning . 
pred:    you couldn do nothing more than learning .
```

## Reference

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
