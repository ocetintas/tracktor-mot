import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import nms, box_iou


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect, thresholds, device='cuda:0'):
		self.obj_detect = obj_detect

		self.tracks = []
		self.inactive_tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

		self.sigma_active = thresholds['sigma_active']
		self.lambda_active = thresholds['lambda_active']
		self.lambda_new = thresholds['lambda_new']

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
		new_boxes = new_boxes.view(-1, 4)
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
			box = self.tracks[0].box.view(1, 4)
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0)
		return box

	def get_results(self):
		return self.results

	def get_scores(self):
		"""
		Return the scores of the tracks
		"""
		return torch.tensor([t.score for t in self.tracks])

	def motion_box(self, track):
		"""
		Calculate the approximate position of next bounding box assuming constant velocity
		"""
		prev_center = torch.tensor([track.prev_box[0]+track.prev_box[2]/2, track.prev_box[1]+track.prev_box[3]/2])
		center = torch.tensor([track.box[0]+track.box[2]/2, track.box[1]+track.box[3]/2])
		motion = center - prev_center
		next_box = torch.tensor([track.box[0]+motion[0], track.box[1]+motion[1], track.box[2], track.box[3]])
		return next_box

	def get_motion_boxes(self):
		"""
		Get the positions of all tracks but use motion_boxes if available
		"""
		if len(self.tracks) == 1:
			if self.tracks[0].prev_box is None:
				box = self.tracks[0].box.view(1, 4)
			else:
				box = self.motion_box(self.tracks[0]).view(1, 4)
		elif len(self.tracks) > 1:
			motion_boxes = []
			for t in self.tracks:
				if t.prev_box is None:
					motion_boxes += [t.box]
				else:
					motion_boxes += [self.motion_box(t)]
			box = torch.stack(motion_boxes, 0)
		else:
			box = torch.zeros(0)
		return box

	def move_to_inactives(self, inactives):
		"""
		Move inactive tracks and update tracks
		"""
		self.inactive_tracks += inactives
		self.tracks = [t for t in self.tracks if t not in inactives]

	def data_association(self, frame):
		"""
		Data association is performed with bounding box regression from detections of the previous frame
		"""
		# Only if there are existing tracks
		if self.tracks:
			# Bounding box regression of previous detections
			reg_boxes, reg_scores = self.obj_detect.bbox_regression(frame['img'], self.get_motion_boxes())

			# Update current track information and move tracks with low scores to inactives
			inactives = []
			for i, t in enumerate(self.tracks):
				# Update tracks with scores above the threshold
				if reg_scores[i] > self.sigma_active:
					self.tracks[i].prev_box = self.tracks[i].box
					self.tracks[i].score = reg_scores[i]
					self.tracks[i].box = reg_boxes[i]
				# Move low scores to inactives
				else:
					inactives += [t]
			self.move_to_inactives(inactives)

			# NMS of regressed boxes
			inactives = []
			i_keep = nms(self.get_pos().to(self.device), self.get_scores().to(self.device), self.lambda_active)
			for i, t in enumerate(self.tracks):
				if i not in i_keep:
					inactives += [t]
			self.move_to_inactives(inactives)

	def find_new_tracks(self, frame):
		"""
		Create new tracks if the detected object is not in any of the active tracks
		"""

		# Object detection with the current frame
		new_boxes, new_scores = self.obj_detect.detect(frame['img'])

		# Filter out low scores
		i_keep = torch.gt(new_scores, self.sigma_active).nonzero().view(-1)
		new_boxes = new_boxes[i_keep]
		new_scores = new_scores[i_keep]

		# NMS of new detections
		i_keep = nms(new_boxes.view(-1, 4).to(self.device), new_scores.to(self.device), self.lambda_new)
		new_boxes = new_boxes[i_keep]
		new_scores = new_scores[i_keep]

		# Start a new track if the iou with all existing tracks is below the threshold
		if self.tracks and torch.numel(new_boxes):
			iou = box_iou(self.get_pos(), new_boxes.view(-1, 4))
			iou_bool = torch.gt(iou, self.lambda_new)
			iou_bool = iou_bool.any(dim=0) 	# Check if any iou with previous tracks is above the threshold
			i_keep = (iou_bool == False).nonzero().view(-1) 	# Only accept if no iou is greater than the threshold
			new_boxes = new_boxes[i_keep]
			new_scores = new_scores[i_keep]
		self.add(new_boxes, new_scores)

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""

		# Data association with bounding box regression
		self.data_association(frame)

		# Find new tracks in the current frame
		self.find_new_tracks(frame)

		# Results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])
		self.im_index += 1


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box
		self.score = score

		self.prev_box = None
