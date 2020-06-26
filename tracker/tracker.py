import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect, thresholds, device='cuda:0'):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

		# --------------------------
		self.inactive_tracks = []
		self.score_det = thresholds['score_det']
		self.nms_det = thresholds['nms_det']
		self.nms_reg = thresholds['nms_reg']

		self.device = device

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
				new_scores[i],
				self.track_num + i
			))
		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self):
		"""
		Find tracks in the current frame
		"""
		if self.tracks:
			self.bbox_regression_tracks()


	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# Data association with bounding box regression
		self.obj_detect.cache(frame['img'])
		self.data_association()

		# Object detection
		boxes, scores = self.obj_detect.detect(frame['img'])
		self.find_new_tracks(boxes, scores)

		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1

	def get_results(self):
		return self.results

	def get_scores(self):
		"""
		Return the scores of the tracks
		"""
		return torch.tensor([t.score for t in self.tracks])

	def add_inactives(self, inactives):
		"""
		Move inactive tracks from active tracks list to inactive tracks list
		"""
		self.tracks = [t for t in self.tracks if t not in inactives]
		self.inactive_tracks += inactives

	def bbox_regression_tracks(self):
		"""
		Bounding box regression from previous detections, followed by NMS
		"""
		# Bounding box regression of previous detections
		boxes = self.get_pos()
		new_boxes, new_scores = self.obj_detect.bbox_regression(boxes)
		new_boxes = clip_boxes_to_image(new_boxes, self.obj_detect.im_size)

		# Update current tracks and move low scores to inactives
		inactives = []
		for i, t in enumerate(self.tracks):
			# Above the detection threshold
			if new_scores[i] > self.score_det:
				self.tracks[i].score = new_scores[i]
				self.tracks[i].box = new_boxes[i]
			else:
				inactives += [t]
		self.add_inactives(inactives)

		# NMS
		inactives = []
		i_keep = nms(self.get_pos().to(self.device), self.get_scores().to(self.device), self.nms_reg)
		for i, t in enumerate(self.tracks):
			if i not in i_keep:
				inactives += [t]
		self.add_inactives(inactives)

	def find_new_tracks(self, boxes, scores):
		"""
		Create new tracks that does not exist in the previous frame
		"""
		# Filter low scores
		i_keep = torch.gt(scores, self.score_det).nonzero().view(-1)
		boxes = boxes[i_keep]
		scores = scores[i_keep]

		# NMS within detections
		i_keep = nms(boxes.to(self.device), scores.to(self.device), self.nms_det)
		boxes = boxes[i_keep]
		scores = scores[i_keep]

		# Compare with old tracks
		for t in self.tracks:
			temp_boxes = torch.cat([t.box.view(1, -1).to(self.device), boxes])
			temp_scores = torch.cat([torch.tensor([10.0]).to(self.device), scores])
			i_keep = nms(temp_boxes.to(self.device), temp_scores.to(self.device), self.nms_det)
			i_keep = i_keep[torch.ge(i_keep, 1)] - 1
			boxes = boxes[i_keep]
			scores = scores[i_keep]
		self.add(boxes, scores)


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box
		self.score = score
