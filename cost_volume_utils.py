#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:55:45 2021

@author: laurent
"""
import matplotlib.pyplot as plt

def exploreCostVolume(costVol) :
	
	fig, ax = plt.subplots()
	ax.volume = costVol
	ax.index = costVol.shape[2] // 2
	ax.imshow(costVol[:,:,ax.index])
	fig.canvas.mpl_connect('key_press_event', process_key)
	plt.show()

def process_key(event):
	fig = event.canvas.figure
	ax = fig.axes[0]
	if event.key == 'p':
		previous_slice(ax)
	elif event.key == 'n':
		next_slice(ax)
	
	fig.canvas.set_window_title('Slice #{}'.format(ax.index))
	fig.canvas.draw()

def previous_slice(ax):
	"""Go to the previous slice."""
	volume = ax.volume
	ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
	ax.images[0].set_array(volume[:,:,ax.index])

def next_slice(ax):
	"""Go to the next slice."""
	volume = ax.volume
	ax.index = (ax.index + 1) % volume.shape[2]
	ax.images[0].set_array(volume[:,:,ax.index])