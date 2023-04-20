import torch
import train
import models.bert as bert
import json

def init_model():
    bert_model = bert.Bert(
        vocab_size=train.HYPER_PARAM['vocab_size'],
        d_emb=train.HYPER_PARAM['d_emb'],
        n_head=train.HYPER_PARAM['n_head'],
        n_layers=train.HYPER_PARAM['n_layers'],
        hidden_dim=train.HYPER_PARAM['hidden_dim'],
        max_len=train.HYPER_PARAM['max_len'],   
        dropout=train.HYPER_PARAM['dropout']
    )
    model = bert.BertForSequenceClassification(
        bert_model,
        train.HYPER_PARAM['vocab_size'],
        train.HYPER_PARAM['d_emb']
    )

    checkpoint = torch.load(train.HYPER_PARAM['model_path'] +'/ddp_final.pt')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    bert_model.eval()
    return model, bert_model

def inference(s, model, bert_model, token2idx, idx2token, cat_bos_eos=True):
    tokens = str2token(s, token2idx)
    if cat_bos_eos:
        tokens = [token2idx['<bos>']] + tokens + [token2idx['<eos>']]
    mask_ids = [idx for idx, token in enumerate(tokens) if token == token2idx['<mask>']]
    tokens = torch.tensor(tokens).unsqueeze(0)
    seg = torch.zeros_like(tokens, dtype=torch.long) + 1
    lens = torch.tensor([tokens.shape[1]], dtype=torch.long)
    reps = bert_model(tokens.cuda(), seg.cuda(), lens.cuda())
    is_next_pred, classification_pred = model(tokens.cuda(), seg.cuda(), lens.cuda())
    classification_pred = torch.argmax(classification_pred.transpose(-1, -2), dim=-1)
    for idx in mask_ids:
        tokens[:,idx] = classification_pred[:,idx]
    if cat_bos_eos:
        tokens = tokens[:, 1:-1]
    sent = token2str(tokens, idx2token)
    return sent, is_next_pred.cpu(), reps.cpu()

def str2token(s, token2idx):
    tokens = []
    s = s.lower()
    s = s.split()
    for c in s:
        tokens.append(token2idx.get(c, token2idx['<unk>']))
    return tokens

def token2str(tokens, idx2token):
    s = []
    tokens = tokens.squeeze().tolist()
    for token in tokens:
        s.append(idx2token[str(token)])
    return ' '.join(s)

# load from json file
def load_token_dump(token_dump_file):
    tokens_dump = json.load(open(token_dump_file, 'r'))
    token2idx = tokens_dump['token2idx']
    idx2token = tokens_dump['idx2token']
    tokens = tokens_dump['tokens']
    return token2idx, idx2token, tokens

if __name__ == "__main__":
    #token_dump_file = 'token_dump.json'
    token2idx, idx2token, tokens = load_token_dump(train.HYPER_PARAM['tokens_dump_file'])
    model, bert_model = init_model()
    model = model.cuda()
    bert_model = bert_model.cuda()
    print("input a sentence and randomly mask some word by <mask>")
    while True:
        print("\n")
        print("sent:", end='\t')
        s = input()
        sent, is_next_pred, pres = inference(s, model, bert_model, token2idx, idx2token, cat_bos_eos=True)
        print("pred:\t", sent)
    
