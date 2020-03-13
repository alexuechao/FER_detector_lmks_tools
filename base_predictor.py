# Predictor of face
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import copy
import logging
import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)
import h5py
import skimage.io
import cv2
import datetime
import glob
from Retinaface.retinaface import RetinaFace

class DetectionPredictor():
	def __init__(self, config):
		self.config = config
		self.roidb = None
		self.save_file = open(self.config.save, 'w')

	def make_roidb(self, images):
		roidb = []
		for i in range(len(images)):
			roi_rec = dic()
			roi_rec['image'] = image[i]
			roidb.append(roi_rec)
		self.roidb = roidb

	def check_image_path(self, image_path):
		support_format = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
		if  image_path.split('.')[-1] in support_format:
			return True
		else:
			return False

	def get_all_images(self):
		all_images = []

		if not self.config.file == '':
			with open(self.config.file,'r') as f:
				lines = f.readlines()
			for line in lines:
				image_path = line.strip().split(' ')[0]
				if self.check_image_path(image_path):
					all_images.append(image_path)

		if not self.config.dir == '':
			image_walk_list = os.walk(self.config.dir)
			for one in image_walk_list:
				for two in one[-1]:
					image_path = os.path.join(one[0], two)
					if self.check_image_path(image_path):
						all_images.append(image_path)
		self.all_images = all_images

	def get_batch_images_list(self):
		step = self.config.batch_images_num
		batch_images_list = [self.all_images[i:i+step]
		                     for i in range(0,len(self.all_images),step)]
		self.batch_images_list = batch_images_list

	def get_save_string(self, image_path, rects,lmks):
		save_string = ''
		save_string += image_path
		save_string += ' '
		save_string += (' '.join('{:.4f}'.format(rects[rect_i])
		                         for rect_i in range(4)))
		save_string += ' '
		save_string += (' '.join('{:.4f} {:.4f}'.format(lmks[l][0], lmks[l][1]) for l in range(5)))
		save_string += '\n'
		return save_string

	def get_predictor(self):
		detector = RetinaFace(prefix = self.config.prefix,
		                        epoch = self.config.epoch,
		                        ctx_id = self.config.ctx_id,
		                        network = 'net3')
		return detector

	def get_boxes(self, image, pyramid = False):
		detector = self.get_predictor()
		if not pyramid:
			do_flip = False
			target_size = 1024
			max_size = 1980
			im_shape = image.shape
			im_size_min = np.min(im_shape[0:2])
			im_size_max = np.min(im_shape[0:2])
			im_scale = float(target_size) / float(im_size_min)
			if np.round(im_scale * im_size_max) > max_size:
				im_scale = float(max_size)/ float(im_size_max)
			scales = [im_scale]
		else:
			do_flip = True
			TEST_SCALES = [500,800,1100,1400,1700]
			target_size = 800
			max_size = 1200
			im_shape = image.shape
			im_size_min = np.min(im_shape[0:2])
			im_size_max = np.min(im_shape[0:2])
			im_scale = float(target_size) / float(im_size_min)
			if np.round(im_scale * im_size_max) > max_size:
				im_scale = float(max_size)/ float(im_size_max)
			scales = [float(scale)/target_size*im_scale for scale in TEST_SCALES]
		boxes, landmarks = detector.detect(image, threshold = self.config.threshold,
			                               scales = scales, do_flip = do_flip)
		return boxes, landmarks

	def predict(self):
		self.get_all_images()
		for i, image_path in enumerate(self.all_images):
			print(image_path)
			image = cv2.imread(image_path)
			boxes, landmarks = self.get_boxes(image, pyramid = False)
			print(boxes)
			print(landmarks)
			landmarks = landmarks.tolist()
			for idx in self.select_boxes_index(select_type=self.config.select_type, boxes=boxes, top_k=1,img_path=image_path):
				rects = list(boxes[idx])
				lmks = list(landmarks[idx])
				#import pdb
				#pdb.set_trace()
				if ((rects[2] - rects[0]) * (rects[3] - rects[1])) <=1:
					continue
				info_instance = dict()
				info_instance['image_path'] = image_path
				info_instance['rects'] = rects
				info_instance['lmks'] = lmks
				save_string = self.get_save_string(info_instance['image_path'],
					                               info_instance['rects'],
					                               info_instance['lmks'])
				self.save_file.write(save_string)

	def select_boxes_index(self, select_type, boxes, top_k, img_path=None):
		if select_type == 'all':
		    return range(boxes.shape[0])
		elif select_type == 'score':
		    score = boxes[:, -1]
		    score_index = np.argsort(score)[::-1]
		    if boxes.shape[0] < top_k:
		        score_index = score_index[:boxes.shape[0]]
		    else:
		        score_index = score_index[top_k]
		    return score_index
		elif select_type == 'size':
		    size = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
		    size_index = np.argsort(size)[::-1]
		    if boxes.shape[0] < top_k:
		        size_index = size_index[:boxes.shape[0]]
		    else:
		        size_index = size_index[:top_k]
		    return size_index
		elif select_type == 'center':
		    img_h, img_w = cv2.imread(img_path).shape[:2]
		    center = [img_w/2., img_h/2.]
		    lenght_list = []
		    for i in range(boxes.shape[0]):
		        x = (boxes[i, 0] + boxes[i, 2]) / 2.
		        y = (boxes[i, 1] + boxes[i, 3]) / 2.
		        lenght = np.sqrt(
		            np.power(center[0] - x, 2) + np.power(center[1] - y, 2))
		        lenght_list.append(lenght)
		    lenght_index = np.argsort(lenght_list)
		    if boxes.shape[0] < top_k:
		        lenght_index = lenght_index[:boxes.shape[0]]
		    else:
		        lenght_index = lenght_index[:top_k]
		    return lenght_index
		else:
		    assert False, 'not support select_type: {}'.format(select_type)

def parse_args():
    parser = argparse.ArgumentParser(description='Get rects and lmks')
    parser.add_argument('--file', help='input image file', default='', type=str)
    parser.add_argument('--dir', help='input image dir', default='', type=str)
    parser.add_argument('--save', help='save result file.', required=False, type=str)
    parser.add_argument('--prefix',help='input the model',default='./Retinaface/mnet.25/mnet.25',type=str)
    parser.add_argument('--epoch', help='model epoch', default=0, type=int)
    parser.add_argument('--ctx_id', help='gpu id', default=-1, type=int)
    parser.add_argument('--threshold', help='the detect threshold', default=0.85, type=int)
    parser.add_argument('--select_type', help='output rects type', default='all', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_args()
	det_predictor = DetectionPredictor(args)
	det_predictor.predict()