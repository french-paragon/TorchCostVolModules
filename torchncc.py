#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:25:50 2021

@author: laurent
"""

import torch
import torch.nn as nn
import numpy as np

class NccCostVolume(nn.Module) :
	"""
	A module to construct a cost volume using NCC
	"""
	
	def __init__(self, 
				  correlationWidth, 
				  horizontalCorrelationRadius = 3, 
				  verticalCorrelationRadius=3,
				  padToSame = True,
				  normalizationEpsilon = 1e-9,
				  compressor = None) :
		super().__init__()
		
		self.padToSame = padToSame
		self.normalizationEpsilon = normalizationEpsilon
		
		self.correlationWidth = correlationWidth
		
		self.correlationWidth = correlationWidth
		
		self.horizontalCorrelationRadius = horizontalCorrelationRadius
		self.verticalCorrelationRadius = verticalCorrelationRadius
	
		self.kernelSize = (2*verticalCorrelationRadius+1, 2*horizontalCorrelationRadius+1)
	
		pTup = 0
		if padToSame :
			pTup = (horizontalCorrelationRadius, verticalCorrelationRadius)
	
		self.unfoldlayer = nn.Unfold(self.kernelSize, padding=pTup)
		self.padLeft = nn.ZeroPad2d((0, self.correlationWidth-1, 0, 0))
		self.compressor = compressor
	
	def forward(self, im_left, im_right) :
		
		b, c, h, w_left = tuple(im_left.shape)
		b_r, c_r, h_r, w_right = tuple(im_right.shape)
		
		nh = h if self.padToSame else h - 2*self.verticalCorrelationRadius
		nw_l = (w_left if self.padToSame else w_left - 2*self.verticalCorrelationRadius) + self.correlationWidth-1
		nw_r = w_right if self.padToSame else w_right - 2*self.verticalCorrelationRadius
		
		#padLeft
		ext_patches_l = self.padLeft(im_left)
		
		#unfold the patches
		patches_l = self.unfoldlayer(ext_patches_l)
		patches_r = self.unfoldlayer(im_right)
		
		if self.compressor is not None :
			patches_l = torch.matmul(self.compressor, patches_l)
			patches_r = torch.matmul(self.compressor, patches_r)
		
		#subtract the mean
		patches_l -= patches_l.mean(1, keepdim=True)
		patches_r -= patches_r.mean(1, keepdim=True)
		
		#divide by standard deviation
		patches_l /= torch.sqrt(patches_l.square().sum(1, keepdim=True) + self.normalizationEpsilon)
		patches_r /= torch.sqrt(patches_r.square().sum(1, keepdim=True) + self.normalizationEpsilon)
		
		#reshape to have image dimensions in the end
		patches_l = patches_l.reshape(b,-1,nh,nw_l)
		patches_r = patches_r.reshape(b,-1,nh,nw_r)
		
		#do correlation
		viewLeft = patches_l.unfold(-1,self.correlationWidth,1)
		viewLeft = torch.moveaxis(viewLeft,1,4)
		
		viewRight = torch.moveaxis(patches_r,1,3)
		viewRight = viewRight[...,np.newaxis]
		
		costVolume = torch.matmul(viewLeft, viewRight).squeeze(-1)
		
		return costVolume