# FLIP models template

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

__all__ = ['FLIP_Discriminator', 'FLIP_Link_prediction', 'SkipGramModel']


def xavier_init(m):
	""" Xavier initialization """
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0)


class FLIP_Discriminator(nn.Module):
	def __init__(self, emb_dim):
		super(FLIP_Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(2*emb_dim, emb_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(emb_dim, 1),
			# nn.Sigmoid(),
		)
		self.model.apply(xavier_init)

	def forward(self, u, v):
		link_embd = torch.cat((u, v), 1)
		y_hat = self.model(link_embd)
		return y_hat
	
	
class FLIP_Link_prediction(nn.Module):
	def __init__(self, emb_dim):
		super(FLIP_Link_prediction, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(2*emb_dim, emb_dim),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(emb_dim, 1),
			# nn.Sigmoid(),
		)
		self.model.apply(xavier_init)

	def forward(self, u, v):
		link_embd = torch.cat((u, v), 1)
		y_hat = self.model(link_embd)
		return y_hat


class SkipGramModel(nn.Module):

	def __init__(self, emb_size, emb_dim, sparse):
		super(SkipGramModel, self).__init__()
		self.emb_size = emb_size
		self.emb_dim = emb_dim

		self.u_embeddings = nn.Embedding(emb_size, emb_dim, sparse=sparse)
		self.v_embeddings = nn.Embedding(emb_size, emb_dim, sparse=sparse)

		initrange = 1.0 / self.emb_dim
		init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
		init.constant_(self.v_embeddings.weight.data, 0)


	def forward(self, pos_u, pos_v, neg_v):
		emb_u = self.u_embeddings(pos_u)
		emb_v = self.v_embeddings(pos_v)
		emb_neg_v = self.v_embeddings(neg_v)

		score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
		score = torch.clamp(score, max=10, min=-10)
		score = -F.logsigmoid(score)

		neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
		neg_score = torch.clamp(neg_score, max=10, min=-10)
		neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

		return torch.mean(score + neg_score)


	def return_embedding(self, *args):
		out = list()
		for arg in args:
			# import pdb; pdb.set_trace()
			embedded_arg = self.u_embeddings.weight.data[arg.long()]
			out.append(embedded_arg)
			
		# embedded_u = self.u_embeddings.weight.data[u]
		# embedded_v = self.u_embeddings.weight.data[v]
		return out


	def save_embedding(self, id2node, file_name):
		embedding = self.u_embeddings.weight.cpu().data.numpy()
		with open(file_name, 'w') as f:
			f.write('%d %d\n' % (len(id2node), self.emb_dim))
			for wid, w in id2node.items():
				e = ' '.join(map(lambda x: str(x), embedding[wid]))
				f.write('%s %s\n' % (str(w), e))