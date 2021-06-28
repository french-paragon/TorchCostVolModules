#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:18:30 2021

@author: laurent
"""

import torch

def Pix17R3Filter() :
	
	matr = torch.zeros(17,7,7)
	
	#center pixel
	matr[0,3,3] = 1.
	
	#four secondary superpixels
	matr[1,1:3,3] = 0.5
	matr[2,4:6,3] = 0.5
	matr[3,3,1:3] = 0.5
	matr[4,3,4:6] = 0.5
	
	#four tertiary superpixels
	matr[5,1:3,1:3] = (1./3.)
	matr[5,1,1] = 0.0
	
	matr[6,4:6,4:6] = (1./3.)
	matr[6,5,5] = 0.0
	
	matr[7,4:6,1:3] = (1./3.)
	matr[7,5,1] = 0.0
	
	matr[8,1:3,4:6] = (1./3.)
	matr[8,1,5] = 0.0
	
	#four quaternary superpixels
	matr[9,2:5,0] = (1./3.)
	matr[10,2:5,6] = (1./3.)
	matr[11,0,2:5] = (1./3.)
	matr[12,6,2:5] = (1./3.)
	
	#four corner superpixels
	matr[13,0:2,0:2] = 0.25
	matr[14,0:2,5:7] = 0.25
	matr[15,5:7,0:2] = 0.25
	matr[16,5:7,5:7] = 0.25
	
	return matr.reshape(1,17,49)

def Pix17R4Filter() :
	
	matr = torch.zeros(17,9,9)
	
	#center pixel
	matr[0,4,4] = 1.
	
	#four secondary superpixels
	matr[1,2:4,4] = 0.5
	matr[2,5:7,4] = 0.5
	matr[3,4,2:4] = 0.5
	matr[4,4,5:7] = 0.5
	
	#four tertiary superpixels
	matr[5,2:4,2:4] = 0.25
	matr[6,5:7,5:7] = 0.25
	matr[7,5:7,2:4] = 0.25
	matr[8,2:4,5:7] = 0.25
	
	#four quaternary superpixels
	matr[9,3:6,0:2] = (1./6.)
	matr[10,3:6,7:9] = (1./6.)
	matr[11,0:2,3:6] = (1./6.)
	matr[12,7:9,3:6] = (1./6.)
	
	#four corner superpixels
	matr[13,0:2,0:2] = 0.125
	matr[13,2,0:2] = 0.125
	matr[13,0:2,2] = 0.125
	
	matr[14,0:2,7:9] = 0.125
	matr[14,0:2,6] = 0.125
	matr[14,2,7:9] = 0.125
	
	matr[15,7:9,0:2] = 0.125
	matr[15,6,0:2] = 0.125
	matr[15,7:9,2] = 0.125
	
	matr[16,7:9,7:9] = 0.125
	matr[16,6,7:9] = 0.125
	matr[16,7:9,6] = 0.125
	
	return matr.reshape(1,17,81)
