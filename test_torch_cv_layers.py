#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:49:43 2021

@author: laurent
"""


import torch

import numpy as np
import lexrtools_pyreadExrChannels as rexr

from cost_volume_utils import exploreCostVolume
from torchncc import NccCostVolume
from torchssd import SsdCostVolume
from torchsad import SadCostVolume

from superpixelfilters import Pix17R3Filter, Pix17R4Filter

import argparse as args
import time

if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Test the ncc torch layer')
	
	parser.add_argument('filename', help="Path to the exr file with appropriate passes.")
	
	parser.add_argument('-w', '--width', type=int, default=40, help="Width of the convolution")
	parser.add_argument('-r', '--radius', type=int, default=3, help="Radius of the square windows for cross-correlations")
	parser.add_argument('--rgb', action='store_true', help='use the color layer instead of the NIR layer')
	parser.add_argument('--method', choices=['ncc', 'ssd', 'sad'], default='ncc', help="Method used for cost volume building.")
	
	parser.add_argument('--cuda', action='store_true', help='perform computations on gpu if available')
	parser.add_argument('--compress', action='store_true', help='use a compression layer')
	
	args = parser.parse_args()
	
	usecuda = torch.cuda.is_available() and args.cuda
	
	if args.rgb :
		left = rexr.readExrLayer(args.filename, 'Center.Color')[:,:,::-1]
		right = rexr.readExrLayer(args.filename, 'Center.Color')[:,:,::-1]
	else :
		left = rexr.readExrChannel(args.filename, 'Left.SimulatedNir.A')[:,:,np.newaxis]
		right = rexr.readExrChannel(args.filename, 'Right.SimulatedNir.A')[:,:,np.newaxis]
		
	batch_left = np.moveaxis(left[np.newaxis,...], 3, 1)
	batch_right = np.moveaxis(right[np.newaxis,...], 3, 1)
		
	batch_left = torch.from_numpy(batch_left)
	batch_right = torch.from_numpy(batch_right)
	
	if usecuda:
		batch_left = batch_left.cuda()
		batch_right = batch_right.cuda()
	
	print(batch_left.shape)
	print(batch_right.shape)
	
	CvLayer = None
	Compressor = None
	
	if args.compress :
		if args.radius == 3 :
			Compressor = Pix17R3Filter()
		if args.radius == 4 :
			Compressor = Pix17R4Filter()
			
		if usecuda:
			Compressor = Compressor.cuda()
	
	if args.method == "ncc" :
		CvLayer = NccCostVolume(args.width, args.radius, args.radius, compressor = Compressor)
	elif args.method == "ssd" :
		CvLayer = SsdCostVolume(args.width, args.radius, args.radius)
	elif args.method == "sad" :
		CvLayer = SadCostVolume(args.width, args.radius, args.radius)
	
	if usecuda:
		CvLayer.cuda()
	
	costVolume = None
	
	with torch.no_grad() :
		
		if usecuda:
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			
			start.record()
			costVolume = CvLayer(batch_left, batch_right)
			end.record()
			
			torch.cuda.synchronize()
			
			print("--- {:.2f} ms ---".format(start.elapsed_time(end)))
			
			costVolume = costVolume.squeeze().cpu().numpy()
			
		else :
			start_time = time.time()
			costVolume = CvLayer(batch_left, batch_right).squeeze().cpu().numpy()
			print("--- {:.2f} ms ---".format((time.time() - start_time)*1000))
		
	print(costVolume.shape)
	
	exploreCostVolume(costVolume)