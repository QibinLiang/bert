"""
    This project reimplments the BERT model(https://arxiv.org/pdf/1810.04805.pdf) and 
    train the model on the wikitext2 dataset. In this project, we do some modification on the
    original BERT model. 
    We use give a probability to the hot tokens to be masked because we find that in such a small 
    dataset, the model will trend to generate the hot tokens e.g. 'I want <mask>' -> 'the the the'
    
"""


# TODO :
# 1. load the model from the checkpoint as
#       well as load the optimizer and the scheduler
# 2. implement the distributed training
# 3. load the hyperparamer from the yaml file
# 4. implement the tensorboard
# 5. implement the early stopping
# 6. refector the valid and the test function to
#       make the function well organized
# 7. add document
# 8. add parser for the command line

import os
import tqdm
import json
import torch
import argparse
# import train_tools
import models.bert as bert
import utils.utils as utils
import utils.dataset as dataset
from torch.utils.data.distributed import DistributedSampler

# we set the idx of <pad> to 0 and <unk> to 1
# and <bos> to 2 and <eos> to 3 by default
HYPER_PARAM = {
    # if ddp is used
    "ddp": False,
    "ddp_backend": "nccl",
    "world_size": 4,
    "ddp_init_method": "FILE",
    "visible_devices": "0,1,2,3",
    # params for tokenization
    "vocab_size": 30005,
    "minimal_frequency": 1,
    "tokens_dump_file": 'data/tokens_dump.json',
    # params for dataset
    "mask_rate": 0.2,
    "max_len": 256,
    "train_data_file": r'data/train_data.txt',
    "valid_data_file": r'data/valid_data.txt',
    "test_data_file": r'data/test_data.txt',
    # params for model
    "d_emb": 768,
    "n_head": 12,
    "n_layers": 12,
    "hidden_dim": 768,
    "dropout": 0.10,
    # params for training
    "batch_size": 6,
    "lr": 3e-5,
    "epochs": 1000,
    "grad_cum": 5,
    "lr_warmup_epoch": 3,
    "lr_scheduler_epoch": 10,
    "device": torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu'),
    # checkpoint
    "model_path": 'checkpoints',
    "load_checkpoint": False,
}


def arg_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--init_file", default=None)
    args = parser.parse_args()
    return args


def prepare_data(ddp=False):
    # 1. load dataset
    # Load the token2idx dictionary
    tokens_dump = json.load(open('data/tokens_dump.json', 'r'))
    token2idx = tokens_dump['token2idx']
    tokens = tokens_dump['tokens']
    hot_tokens = tokens_dump['hot_tokens']
    # Load the dataset
    train_dataset = dataset.TextDataset(
        HYPER_PARAM['train_data_file'], token2idx, tokens=tokens,
        hot_tokens=hot_tokens, max_len=HYPER_PARAM['max_len'], mask_rate=HYPER_PARAM['mask_rate'])
    valid_dataset = dataset.TextDataset(
        HYPER_PARAM['valid_data_file'], token2idx, tokens=tokens,
        hot_tokens=hot_tokens, max_len=HYPER_PARAM['max_len'], mask_rate=HYPER_PARAM['mask_rate'])
    test_dataset = dataset.TextDataset(
        HYPER_PARAM['test_data_file'], token2idx, tokens=tokens,
        hot_tokens=hot_tokens, max_len=HYPER_PARAM['max_len'], mask_rate=HYPER_PARAM['mask_rate'])
    if ddp:
        ddpsampler = DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=HYPER_PARAM['batch_size'],
            num_workers=4,
            collate_fn=dataset.collate_fn_pad,
            sampler=ddpsampler
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=HYPER_PARAM['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn_pad,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=HYPER_PARAM['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn_pad)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=HYPER_PARAM['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn_pad)
    return train_loader, valid_loader, test_loader


def build_models():
    bert_model = bert.Bert(
        vocab_size=HYPER_PARAM['vocab_size'],
        d_emb=HYPER_PARAM['d_emb'],
        max_len=HYPER_PARAM['max_len'],
        n_head=HYPER_PARAM['n_head'],
        n_layers=HYPER_PARAM['n_layers'],
        hidden_dim=HYPER_PARAM['hidden_dim'],
        dropout=HYPER_PARAM['dropout']
    )

    model = bert.BertForSequenceClassification(
        bert_model,
        HYPER_PARAM["vocab_size"],
        HYPER_PARAM["d_emb"],
    )
    return model


def train(
        model,
        train_loader,
        optimizer,
        scheduler,
        is_next_loss,
        classifier_loss,
        alloc_device
):

    model.train()
    t = tqdm.tqdm(train_loader)
    cum_step = 0
    for _, item in enumerate(t):
        x, y, seg, is_next, x_lens = item
        x, y, seg, is_next, x_lens = x.to(alloc_device), \
            y.to(alloc_device), \
            seg.to(alloc_device), \
            is_next.to(alloc_device), \
            x_lens.to(alloc_device)

        is_next_pred, classification = model(x, seg, x_lens)
        is_next_loss_value = is_next_loss(is_next_pred, is_next)
        classification_loss_value = classifier_loss(classification, y)
        cost = is_next_loss_value + classification_loss_value
        cost.backward()

        # gradient accumulation
        if cum_step % HYPER_PARAM['grad_cum'] == 0:
            optimizer.step()
            optimizer.zero_grad()
        cum_step += 1
        t.set_postfix_str(
            f'loss: {cost.item():.4f},is_next_loss: {is_next_loss_value.item():.4f}, \
            classification_loss: {classification_loss_value.item():.4f}, \
            lr: {optimizer.param_groups[0]["lr"]:.6f}'
        )
    scheduler.step()
    return model, optimizer


def eval(
        model,
        valid_loader,
        is_next_loss,
        classifier_loss,
        alloc_device
):

    model.eval()
    total_cost = 0
    with torch.no_grad():
        for _, item in enumerate(valid_loader):
            x, y, seg, is_next, x_lens = item
            x, y, seg, is_next, x_lens = x.to(alloc_device), \
                y.to(alloc_device), \
                seg.to(alloc_device), \
                is_next.to(alloc_device), \
                x_lens.to(alloc_device)
            is_next_pred, classification = model(x, seg, x_lens)
            is_next_loss_value = is_next_loss(is_next_pred, is_next)
            classification_loss_value = classifier_loss(classification, y)
            cost = is_next_loss_value + classification_loss_value
            total_cost += cost.item()
    return total_cost / len(valid_loader)


def test(
        model,
        test_loader,
        is_next_loss,
        classifier_loss,
        alloc_device
):

    model.eval()
    total_cost = 0
    with torch.no_grad():
        for _, item in enumerate(test_loader):
            x, y, seg, is_next, x_lens = item
            x, y, seg, is_next, x_lens = x.to(alloc_device), \
                y.to(alloc_device), \
                seg.to(alloc_device), \
                is_next.to(alloc_device), \
                x_lens.to(alloc_device)
            is_next_pred, classification = model(x, seg, x_lens)
            is_next_loss_value = is_next_loss(is_next_pred, is_next)
            classification_loss_value = classifier_loss(classification, y)
            cost = is_next_loss_value + classification_loss_value
            total_cost += cost.item()
    return total_cost / len(test_loader)


# save the model
def save_model(
        model,
        optimizer,
        epoch,
        path
):

    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, path)


# load the model
def load_model(
        model,
        optimizer,
        path
):

    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    # model_state = {}
    # for k, v in checkpoint['model'].items():
    #    name = k[7:]
    #    model_state[name] = v
    # model.load_state_dict(model_state)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def main(args):
    load_checkpoint=HYPER_PARAM['load_checkpoint']
    ddp = HYPER_PARAM['ddp']
    backend = HYPER_PARAM['ddp_backend']
    init_method = HYPER_PARAM['ddp_init_method']
    world_size = HYPER_PARAM['world_size']
    if ddp:
        print(args.init_file)
        os.environ['CUDA_VISIBLE_DEVICES'] = HYPER_PARAM['visible_devices']
        torch.distributed.init_process_group(
            backend=backend,
            init_method=args.init_file,
            world_size=world_size,
            rank=args.rank
        )

    # 1. prepare data
    print('preparing data...')
    train_loader, valid_loader, test_loader = prepare_data(ddp)
    # 2. build models
    print('building models...')
    model = build_models()
    # Print the number of parameters for each layer
    total_params = 0
    for name, param in model.named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name:20s}: {layer_params:,d}")
    # Print the total number of parameters
    print(f"\nTotal: {total_params:,d}")
    # 3. build optimizer and loss
    print('building optimizer and loss...')
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAM['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.995)
    # scheduler = utils.GradualWarmupScheduler(
    #    optimizer, multiplier=1,
    #    total_epoch=HYPER_PARAM['lr_warmup_epoch'], after_scheduler=lr_scheduler)
    # scheduler = None
    # is_next_loss = torch.nn.NLLLoss()
    # classifier_loss = torch.nn.NLLLoss()
    is_next_loss = torch.nn.NLLLoss(ignore_index=0)
    classifier_loss = torch.nn.NLLLoss(ignore_index=0)
    if load_checkpoint:
        model, optimizer, epoch = load_model(
            model, optimizer, HYPER_PARAM['model_path'] + '/ddp_final.pt')
    if ddp:
        model.to(args.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        alloc_device = torch.device(args.rank)
        current_rank = args.rank
        # model, alloc_device = train_tools.local_ddp_init(model, args.rank)
    else:
        alloc_device = torch.device(HYPER_PARAM['device'])
        model.to(alloc_device)
        current_rank = 0
    # 4. train
    best_cost = 1e10
    epoch = 0
    # load checkpoint
    print(f'continue training from epoch {epoch}...')
    for epoch in range(HYPER_PARAM['epochs']):
        model, optimizer = train(
            model, train_loader,
            optimizer, scheduler,
            is_next_loss, classifier_loss, alloc_device=alloc_device)
        valid_cost = eval(
            model, valid_loader,
            is_next_loss, classifier_loss, alloc_device=alloc_device)
        print(
            f'epoch: {epoch}, valid_loss: {valid_cost:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')

        if valid_cost < best_cost and current_rank == 0:
            best_cost = valid_cost
            save_model(
                model, optimizer, epoch,
                HYPER_PARAM['model_path'] + '/ddp_final.pt',
            )
            print('best model saved')

        if epoch % 25 == 0 and current_rank == 0:
            save_model(
                model, optimizer, epoch,
                HYPER_PARAM['model_path'] + '/ddp_epoch_{}.pt'.format(epoch)
            )
            print('model saved')

    test_cost = test(
        model, test_loader,
        is_next_loss, classifier_loss, alloc_device=alloc_device)
    print(f'test_loss: {test_cost:.4f}')


if __name__ == "__main__":
    args = arg_options()
    main(arg_options())
