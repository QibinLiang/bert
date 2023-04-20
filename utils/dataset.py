import json
import torch
import numpy
import typing
import random
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """ The dataset for generate the formated data for the model. This dataset will process
    the data in the way mentioned in the BERT paper.

    The input data will be in the form of:
        <bos> x_1 x_2 x_3 ... <mask> ... x_n <eos> y_1 <mask> y_3 ... y_m <eos>
    The label of the input data will be in the form of:
        <pad> <pad> <pad> <pad> ... x_t ... <pad> <pad> <pad> <pad> ... y_t ... <pad>
    The segment of the input data will be in the form of:
        [1, 1, 1, 1, ..., 1, 2, 2, 2, 2, ..., 2]
    The next token will be in the form of:
        [1]

    Args:
        file: the path to the file containing the data
        token2idx: the token2idx dictionary
        tokens: the list of all the tokens
        hot_tokens: the list of the hot tokensss
        max_len: the maximum length of the sequence

    Output:
        data: the input data
        label: the label of the input data
        seg: the segment of the input data
        is_next: the label whether the sentence pair contains the next sentence
    """
    def __init__(
        self, 
        file: str, 
        token2idx: dict, 
        tokens: list, 
        hot_tokens: list, 
        max_len: int = 512,
        mask_rate: float = 0.15
    ) -> None:
        self.token2idx = token2idx
        self.max_len = max_len
        self.mask_rate = mask_rate
        self.data = self.load_data(file)
        self.mask_number = token2idx['<mask>']
        self.all_tokens = tokens
        self.hot_tokens = hot_tokens
        #self.hot_tokens += ['<unk>']

    def load_data(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')
                data.append((line[0], line[1]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx: the index of the data
        Returns:
            x_data: the input data
            x_label: the label of the input data
            x_seg: the segment of the input data
            y_data: the input data
        """
        x, y, is_next = self.get_data_pair(idx)
        x_mask = self.random_mask(len(x))
        y_mask = self.random_mask(len(y))
        x_data, x_label = self.replace_masked_token(x, x_mask)
        y_data, y_label = self.replace_masked_token(y, y_mask)
        
        # finally add the <bos> and <eos> token to the data at the begin 
        # and the end of the sequence respectively.
        x_data = [self.token2idx['<bos>']] + x_data + [self.token2idx['<eos>']]
        x_label = [self.token2idx['<pad>']] + x_label + [self.token2idx['<pad>']]
        x_seg = [1] * len(x_data)
        y_seg = [2] * len(y_data)

        # concatenate the x and y data as well as shrink the length to the max_len
        data = (x_data + y_data)[:self.max_len-1] + [self.token2idx['<eos>']]
        label = (x_label + y_label)[:self.max_len-1] + [self.token2idx['<pad>']]
        seg = (x_seg + y_seg)[:self.max_len-1] + [self.token2idx['<pad>']]
        return data, label, seg, is_next

    # TODO : refine the duplicated code
    def replace_masked_token(self, tokens, masks, hot_word_rate=1):
        # x should be like:
        # [ x_1, x_2, x_3, x_4(keep), x_100(random_sampled), x6, x7, x_mask(masked), ......, x_t ]
        # y should be like:
        # [ 0  , 0  , 0  , x_4      , x_5                  ,   ,   , x_8           ,       , 0   ]
        data, label = [], []
        for token, mask in zip(tokens, masks):
            if mask == 2:
                if token in self.hot_tokens:
                    # if the random value is greater than the hot_word_rate, then we won't mask the token.
                    if numpy.random.rand() > hot_word_rate:
                        data.append(token)
                        label.append(self.token2idx['<pad>'])
                    else:
                        data.append(self.token2idx[numpy.random.choice(self.all_tokens)])
                        label.append(token)
                else:
                    data.append(self.token2idx[numpy.random.choice(self.all_tokens)])
                    label.append(token)
            elif mask == 3:
                if token in self.hot_tokens:
                    if numpy.random.rand() > hot_word_rate:
                        data.append(token)
                        label.append(self.token2idx['<pad>'])
                    else:
                        data.append(self.token2idx['<mask>'])
                        label.append(token)
                else:
                    data.append(self.token2idx['<mask>'])
                    label.append(token)
            else:
                data.append(token)
                label.append(self.token2idx['<pad>'])
        return data, label

    def get_data_pair(self, idx):
        # get the data pair by randomly sample from the corpora. 
        # (50% sample the correct data pair, 50% randomly sample the next sentence)
        x = [self.token2idx.get(token, self.token2idx['<unk>']) for token in self.data[idx][0].split()]
        if numpy.random.rand() < 0.5:
            y = self.data[numpy.random.choice(self.__len__(), 1).astype(numpy.integer)[0]][1]
            y = [self.token2idx.get(token, self.token2idx['<unk>']) for token in y.split()]
            is_next = 0
        else:
            y = self.data[idx][1]
            y = [self.token2idx.get(token, self.token2idx['<unk>']) for token in y.split()]
            is_next = 1
        return x, y, is_next

    def random_mask(self, sent_len):
        mask_prob = numpy.random.rand(sent_len)
        # 0 is the unmasked token
        # 1 is the same token
        # 2 is the random replaced token
        # 3 is the masked token
        mask = numpy.zeros_like(mask_prob)
        mask[mask_prob < self.mask_rate] = 1
        mask[numpy.where(mask_prob / self.mask_rate < 0.9)] = 2
        mask[numpy.where(mask_prob / self.mask_rate < 0.8)] = 3
        return mask

class SNLIDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        token2idx: dict, 
        is_train: bool, 
        max_len=128
    ):
        self.token2idx = token2idx
        self.max_len = max_len
        self.premises, self.hypos, self.labels = snli_preprocessing.read_snli(data_path, is_train)

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypo = self.hypos[idx]
        label = self.labels[idx]
        # convert the premise and hypo to the token index
        premise = [self.token2idx.get(token, self.token2idx['<unk>']) for token in premise.lower().split()]
        hypo = [self.token2idx.get(token, self.token2idx['<unk>']) for token in hypo.lower().split()]
        # pad the premise and hypo
        data = [self.token2idx['<bos>']] + premise + [self.token2idx['<eos>']] + hypo
        data = data[:self.max_len-1] + [self.token2idx['<eos>']]
        seg = [1] * (len(premise) + 2) + [2] * (len(hypo) + 1)
        seg = seg[:self.max_len]
        return data, label, seg

# TODO : refactor the code to make it more readable.
def collate_fn_pad(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([len(input) for input, label, seg, is_next in batch ], dtype=torch.long)
    ## pad the data
    input_batch = [torch.tensor(input, dtype=torch.long) for input, label, seg, is_next in batch ]
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch)
    # pad the label
    label_batch = [torch.tensor(label, dtype=torch.long) for input, label, seg, is_next in batch ]
    label_batch = torch.nn.utils.rnn.pad_sequence(label_batch)
    # pad the segment
    seg_batch = [torch.tensor(seg, dtype=torch.long) for input, label, seg, is_next in batch ]
    seg_batch = torch.nn.utils.rnn.pad_sequence(seg_batch)
    is_next = torch.tensor([is_next for input, label, seg, is_next in batch ], dtype=torch.long)
    ## compute mask
    return input_batch.T,label_batch.T, seg_batch.T, is_next, lengths

def collate_fn_pad_snli(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([len(input) for input, label, seg,  in batch ], dtype=torch.long)
    ## pad the data
    input_batch = [torch.tensor(input, dtype=torch.long) for input, label, seg,  in batch ]
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch)
    # pad the label
    label_batch = torch.tensor([label for input, label, seg, in batch ], dtype=torch.long)
    # pad the segment
    seg_batch = [torch.tensor(seg, dtype=torch.long) for input, label, seg,  in batch ]
    seg_batch = torch.nn.utils.rnn.pad_sequence(seg_batch)
    ## compute mask
    return input_batch.T,label_batch.T, seg_batch.T, lengths    

# testing
if __name__ == "__main__":
    import inference
    # Load the token2idx dictionary
    tokens_dump = json.load(open('data/wiki2/tokens_dump.json', 'r'))
    # Load the dataset
    dataset = TextDataset(
        r"D:\ASR\BERT\data\train_data.txt",
        tokens_dump['token2idx'],
        max_len=512,
        tokens=tokens_dump['tokens'],
        hot_tokens=tokens_dump['hot_tokens'],
        )
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_fn_pad
        )
    # Iterate through the dataloader
    for input_batch, label_batch, seg, x_lens in dataloader:
        print(input_batch)
        print(label_batch)
        print(seg)
        print(x_lens)
        break
