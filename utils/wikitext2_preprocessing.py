import os
import re
import json
import collections
from datasets import config

# TODO :
# 1. update the path of the data without hard-coding.
# 2. provide more parameter for data preprocessing.
TRAIN = 'data/wiki.train.raw'
VALID = 'data/wiki.valid.raw'
TEST = 'data/wiki.test.raw'

# get the corpora of wikitext from the file by using regex
# using regex to match the strings which doesn't begin with `==`
# and doesn't end with `==`


def load_raw_corpora(file):
    pattern = r'^(?![=]).*(?![=]).$'
    raw_corpora = []
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().lstrip()
            if re.match(pattern, line):
                raw_corpora.append(line)
    return raw_corpora


# tokenize the corpora and remove the punctuations except for the period.
def tokenization(raw_corpora, minumal_frequency=1, topk=35000, num_hot_tokens=30):
    tokens = []
    for line in raw_corpora:
        line = line.lower()
        line = re.sub(r'[^a-z0-9.,:?\"]+', ' ', line)
        tokens.extend(line.split())
    # count the frequency of the tokens
    freq = collections.Counter(tokens)
    # sort the tokens by the frequency
    tokens = sorted(freq, key=freq.get, reverse=True)[:topk]
    hot_tokens = tokens[:num_hot_tokens]
    print("hot tokens: ", hot_tokens)
    # remove the tokens which only appear once
    tokens = [token for token in tokens if freq[token] >= minumal_frequency]
    tokens = ['<pad>', '<unk>', '<bos>', '<eos>', '<mask>'] + tokens
    token2idx = {token: idx for idx, token in enumerate(tokens)}
    idx2token = {idx: token for idx, token in enumerate(tokens)}
    return tokens, token2idx, idx2token, hot_tokens

# normalize the corpora


def preprocess(raw_corpora):
    data = []
    for line in raw_corpora:
        line = line.lower()
        line = re.sub(r'[^a-z0-9.,:?\s{<unk>}]+', '', line)
        # split corpora by '.'
        line = (line.rstrip() + ' ').split(' . ')
        # remove the empty string
        line = [x+'.' for x in line if x]
        # pairly add the sentence to the data
        for i in range(len(line) - 1):
            data.append((line[i], line[i + 1]))
    return data

# save the data to the file


def save_data(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line[0] + '\t' + line[1] + '\n')

# save the token list and the token2idx and idx2token to the file


def save_token(tokens, token2idx, idx2token, hot_tokens, file):
    tokens_dump = {
        'tokens': tokens, 'token2idx': token2idx,
        'idx2token': idx2token, 'hot_tokens': hot_tokens}
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(tokens_dump, f)


def run(tokens_dump_file, minumal_frequency=1, topk=35000):
    # load the raw corpora
    train_raw = load_raw_corpora(TRAIN)
    valid_raw = load_raw_corpora(VALID)
    test_raw = load_raw_corpora(TEST)
    # tokenize the raw corpora
    train_tokens, train_token2idx, train_idx2token, hot_tokens =\
        tokenization(train_raw, minumal_frequency, topk, num_hot_tokens=30)
    print("the number of tokens is:" + str(len(train_tokens)))
    save_token(train_tokens, train_token2idx,
               train_idx2token, hot_tokens, tokens_dump_file)
    # normalize the raw corpora
    train_data = preprocess(train_raw)
    valid_data = preprocess(valid_raw)
    test_data = preprocess(test_raw)
    save_data(train_data, 'data/train_data.txt')
    save_data(valid_data, 'data/valid_data.txt')
    save_data(test_data, 'data/test_data.txt')


if __name__ == '__main__':
    tokens_dump_file = 'data/tokens_dump.json'
    run(
        tokens_dump_file,  # the file to save the tokens
        minumal_frequency=1,  # the minimal frequency of the tokens
        topk=30000  # the number of tokens
    )
