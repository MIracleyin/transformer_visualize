import argparse
import math
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from torchtext.vocab import Vocab
from dataset import Corpus
from model import TransformerModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = eval_model.generate_square_subsequent_mask(args.bptt).to(device)
    atte_weight = []
    with torch.no_grad():
      for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = data_loader.get_batch(data_source, i)

        ########################################
        ######Your code here########
        ########################################
        if data.size(0) != args.bptt:
          src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        _, atte = eval_model(data, src_mask)
        atte_weight.append(atte)
    atte_weight = torch.cat(atte_weight, dim=0)
    atte_mat = torch.zeros([atte_weight.shape[0], atte_weight.shape[0]])
    for i in range(atte_weight.shape[0]):
        for j in range(atte_weight.shape[0]):
            atte_mat[i, j] = torch.cosine_similarity(atte_weight[i], atte_weight[j])
    return atte_mat

def display_attention(text, attention):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')

    ax.set_xticklabels([t for t in text], rotation=90)
    ax.set_yticklabels([t for t in text])
    ax.set_title('attention')
    plt.show()


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
    model = torch.load("best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pred_text = "mask " + "The best way of approaching philosophy is to ask a few philosophical questions"
    pred_token = data_loader.tokenizer(pred_text)
    counter1 = Counter()
    for word in pred_token:
        counter1.update(word)
    vocab1 = Vocab(counter1)
    pred_data = data_loader.data_process(pred_token)

    pred_data = data_loader.batchify(pred_data, 1)
    atte_mat = pred_evaluate(model, pred_data)

    print(pred_text)
    display_attention(pred_token, atte_mat)

