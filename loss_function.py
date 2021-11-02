# author:Danyang
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable

class L1_loss(_Loss):
	def __init__(self):
		super(L1_loss, self).__init__()

	def forward(self, alpha, alpha_pred, mask):
		loss = 0
		eps = 1e-6

		for i in range(alpha.shape[0]):
			if mask[i, ...].sum() > 0:
				loss = loss + torch.sum(torch.abs(alpha[i, ...] * mask[i, ...] - alpha_pred[i, ...] * mask[i, ...])) / (
							torch.sum(mask[i, ...]) + eps)
		loss = loss / alpha.shape[0]
		return loss

class compose_loss(_Loss):
	def __init__(self):
		super(compose_loss, self).__init__()

	def forward(self, image, alpha_pred, fg, bg, mask):
		alpha_pred = (alpha_pred + 1) / 2
		comp = fg * alpha_pred + (1 - alpha_pred) * bg
		loss = 0
		eps = 1e-6

		for i in range(image.shape[0]):
			if mask[i, ...].sum() > 0:
				loss = loss + torch.sum(torch.abs(image[i, ...] * mask[i, ...] - comp[i, ...] * mask[i, ...])) / (
						torch.sum(mask[i, ...]) + eps)
		loss = loss / image.shape[0]

		return loss

class gradient_loss(_Loss):
	def __init__(self):
		super(gradient_loss,self).__init__()

	def forward(self,alpha,alpha_pred,mask):

		fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
		fx=fx.view((1,1,3,3))
		fx=Variable(fx.cuda())
		fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
		fy=fy.view((1,1,3,3))
		fy=Variable(fy.cuda())

		G_x = F.conv2d(alpha,fx,padding=1)
		G_y = F.conv2d(alpha,fy,padding=1)
		G_x_pred = F.conv2d(alpha_pred,fx,padding=1)
		G_y_pred = F.conv2d(alpha_pred,fy,padding=1)

		x_loss = 0
		y_loss = 0
		eps = 1e-6

		for i in range(G_x.shape[0]):
			if mask[i, ...].sum() > 0:
				x_loss = x_loss + torch.sum(torch.abs(G_x[i, ...] * mask[i, ...] - G_x_pred[i, ...] * mask[i, ...])) / (
						torch.sum(mask[i, ...]) + eps)
		x_loss = x_loss / G_x.shape[0]

		for i in range(G_y.shape[0]):
			if mask[i, ...].sum() > 0:
				y_loss = y_loss + torch.sum(torch.abs(G_y[i, ...] * mask[i, ...] - G_y_pred[i, ...] * mask[i, ...])) / (
						torch.sum(mask[i, ...]) + eps)
		y_loss = y_loss / G_y.shape[0]

		loss = x_loss+y_loss
		return loss

class GANloss(_Loss):
	def __init__(self):
		super(GANloss,self).__init__()

	def forward(self,pred,label_type):
		MSE=nn.MSELoss()

		total_loss=0
		for i in range(0,len(pred)):
			if label_type:
				labels=torch.ones(pred[i][0].shape)
			else:
				labels=torch.zeros(pred[i][0].shape)
			labels=Variable(labels.cuda())

			total_loss += MSE(pred[i][0],labels)
		loss = total_loss/len(pred)
		return loss