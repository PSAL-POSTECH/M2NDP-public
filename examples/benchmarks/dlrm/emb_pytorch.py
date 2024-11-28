import torch
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingBagModel(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(EmbeddingBagModel, self).__init__()
    # Embedding bag layer
    self.embedding_bag = nn.EmbeddingBag(vocab_size, embed_dim, mode='sum', sparse=True, dtype=torch.float32)
    # self.embedding_bag.weight.data = torch.ones(vocab_size, embed_dim, dtype=torch.float32, device='cuda')

  def forward(self, text, offsets):
    # Apply the embedding bag layer
    embedded = self.embedding_bag(text, offsets)
    # Apply additional layers
    return embedded


MAX_INDEX = 1000000
EMB_DIM = 256
N_LOOKUP = 80
BATCH_SIZE = 256

model = EmbeddingBagModel(MAX_INDEX, EMB_DIM)
model = model.cuda()


def make_input_data(emb_table_id):
  data = []
  with open(os.path.join(os.path.dirname(__file__), f'../../data/kaggle_emb/emb_{emb_table_id}.txt'), 'r') as f:
    batch = 0
    for line in f:
      accesses = line.split(' ')
      accesses = [int(access) for access in accesses[:N_LOOKUP]]
      data += accesses
      batch += 1
      if batch == BATCH_SIZE:
        break
#  return torch.tensor(data, dtype=torch.long, requires_grad=False)
  return torch.tensor(data, dtype=torch.long, requires_grad=False, device='cuda')


offsets = [N_LOOKUP * i for i in range(BATCH_SIZE)]
# print(offsets)
offsets = torch.tensor(offsets, dtype=torch.int32, requires_grad=False, device='cuda')
indices = make_input_data('0_uniform')

model.eval()
with torch.no_grad():
  output = model(indices, offsets)
  # print(output)

