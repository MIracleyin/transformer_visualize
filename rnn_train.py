#main
# coding: utf-8
import math
import torch
import torch.nn as nn
import time
from dataset import Corpus
from model import RNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(epoch, data_loader, criterion, optimizer):
    if isinstance(model, nn.Module):
        model.train()
    total_loss = 0.
    start_time = time.time()
    log_interval = 200
    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        optimizer.zero_grad()
        output, hidden = model(data)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        loss = criterion(output, targets)
        loss.backward()
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
    batch_counter = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)
            output, hidden = model(data)
            output = output.view(output.size(0) * output.size(1), output.size(2))
            loss = criterion(output, targets)
            total_loss += loss
            batch_counter += 1
    return total_loss / batch_counter
if __name__ == '__main__':
    class Argument(object):
        def __init__(self):
            self.epochs = 100
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
                         bptt=args.bptt)

    # bulid your language model here
    model = RNNModel(nvoc=data_loader.get_ntokens(), ninput=64, nhid=64, nlayers=3)
    model.to(device)

    # optimal
    criterion = nn.CrossEntropyLoss()
    lr = 10.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Train Function
    best_val_loss = float("inf")
    best_model = None

    lstm_train_loss_list = []
    lstm_val_loss_list = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(epoch, data_loader, criterion, optimizer)
        lstm_train_loss_list.append(train_loss)
        val_loss = evaluate(model, data_loader, data_loader.val_data)
        lstm_val_loss_list.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
    test_loss = evaluate(best_model, data_loader, data_loader.test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
