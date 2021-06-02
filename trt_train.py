#main
# coding: utf-8
import argparse
import math
import torch
import torch.nn as nn
import time
from dataset import Corpus
from model import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(epoch, data_loader, criterion, optimizer):
    if isinstance(model, nn.Module):
        model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(args.bptt).to(device)
    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        optimizer.zero_grad()
        if data.size(0) != args.bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output, _ = model(data, src_mask)
        loss = criterion(output.view(-1, ntoken), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss  # 每隔两百取一次平均
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_loader.train_data) // args.bptt, scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return math.exp(cur_loss)

def evaluate(eval_model, data_loader, data_source):
    eval_model.eval()
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(args.bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)
            if data.size(0) != args.bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output, _ = eval_model(data, src_mask)
            output_flat = output.view(-1, ntoken)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == '__main__':
    class Argument(object):
        def __init__(self):
            self.epochs = 2
            self.train_batch_size = 20
            self.eval_batch_size = 10
            self.pred_batch_size = 1
            self.bptt = 35
            self.seed = 1234


    args = Argument()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # load data
    data_loader = Corpus(train_batch_size=args.train_batch_size,
                         eval_batch_size=args.eval_batch_size,
                         pred_batch_size=args.pred_batch_size,
                         bptt=args.bptt)

    ntoken = data_loader.get_ntokens()
    model = TransformerModel(ntoken=data_loader.get_ntokens(), ninp=200, nhid=200, nhead=2, nlayers=2, dropout=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Train Function
    best_val_loss = float("inf")
    best_model = None
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(epoch, data_loader, criterion, optimizer)
        train_loss_list.append(train_loss)
        val_loss = evaluate(model, data_loader, data_loader.val_data)
        val_loss_list.append(val_loss)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

            torch.save(best_model, "best.pt")
            # attn_weight = best_model.transformer_encoder.layers[0].self_attn()

        scheduler.step()
    test_loss = evaluate(best_model, data_loader, data_loader.test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)