import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

class Corpus(object):
    def __init__(self, train_batch_size=20, eval_batch_size=10, pred_batch_size=1, bptt=35):
        self.bptt = bptt
        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        txtline = []
        for line in train_iter:
            txtline.append(line)
            counter.update(self.tokenizer(line))
        self.vocab = Vocab(counter)
        train_iter, val_iter, test_iter = WikiText2()
        train_data = self.data_process(train_iter)
        val_data = self.data_process(val_iter)
        test_data = self.data_process(test_iter)
        pred_data = train_data

        self.train_data = self.batchify(train_data, train_batch_size)
        self.val_data = self.batchify(val_data, eval_batch_size)
        self.test_data = self.batchify(test_data, eval_batch_size)
        self.pred_data = self.batchify(pred_data, pred_batch_size)  # 用于单行预测
        self.text = txtline

    def data_process(self, raw_text_iter):
      data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                           dtype=torch.long) for item in raw_text_iter]
      return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Divide the dataset into batch_size parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def get_ntokens(self):
        return len(self.vocab.stoi)