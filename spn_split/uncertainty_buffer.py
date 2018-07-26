from __future__ import division, print_function
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import copy
import time

def gaussian_fit(data, k_upper_bound=20, mode='aic'):
	models = [GaussianMixture(n_components=i).fit(data) for i in range(1, k_upper_bound + 1)]
	scores = [x.aic(data) if mode=='aic' else x.bic(data) for x in models]
	return models[np.argmin(np.array(scores))]

class uncertain_buffer(object):
	def __init__(self, args):
		super(uncertain_buffer, self).__init__()
		self.args = args
		self.predict_length = int(args.uncertain_xyz) * 3 + int(args.uncertain_action) * 2 + int(args.uncertain_pos) + int(args.uncertain_angle) + int(args.uncertain_speed)
		self.feature_length = int(args.uncertain_pos) + int(args.uncertain_angle) + int(args.uncertain_speed)

		self.gt_data = []
		self.condition_data = []
		self.cnt = 0

	def append_condition(self, xyz=None, action=None, pos=None, angle=None, speed=None):
		data = []
		if xyz is not None:
			data += xyz
		if action is not None:
			data.append(action[0])
			data.append(action[1])
		if pos is not None:
			data.append(pos)
		if angle is not None:
			data.append(angle)
		if speed is not None:
			data.append(speed)
		data = np.array(data).reshape(1, -1)

		self.condition_data.append(data)
		if len(self.condition_data) > self.args.uncertain_buffer_size:
			self.condition_data = self.condition_data[-self.args.uncertain_buffer_size:]

	def append_gt(self, pos=None, angle=None, speed=None):
		data = []
		if pos is not None:
			data.append(pos)
		if angle is not None:
			data.append(angle)
		if speed is not None:
			data.append(speed)
		data = np.array(data).reshape(1, -1)

		self.gt_data.append(data)
		if len(self.gt_data) > self.args.uncertain_buffer_size:
			self.gt_data = self.gt_data[-self.args.uncertain_buffer_size:]

	def uncertainty(self, xyz=None, action=None, pos=None, angle=None, speed=None):
		if self.cnt < 100:
			self.cnt += 1
			return 1

		data = []
		if xyz is not None:
			data += xyz
		if action is not None:
			data.append(action[0])
			data.append(action[1])
		if pos is not None:
			data.append(pos)
		if angle is not None:
			data.append(angle)
		if speed is not None:
			data.append(speed)
		condition = np.array(data).reshape(1, -1)

		data = []
		if pos is not None:
			data.append(pos)
		if angle is not None:
			data.append(angle)
		if speed is not None:
			data.append(speed)
		gt = np.array(data).reshape(1, -1)

		joint = np.concatenate([condition, gt], axis=1)

		condition_data = np.concatenate(self.condition_data, axis=0)
		gt_data = np.concatenate(self.gt_data, axis=0)
		joint_data = np.concatenate([condition_data, gt_data], axis=1)

		if self.cnt % self.args.uncertainty_update_freq == 0:
			self.joint_model = gaussian_fit(joint_data)
			self.condition_model = gaussian_fit(condition_data)

		self.cnt += 1

		joint_likelihood = self.joint_model.predict(joint)
		condition_likelihood = self.condition_model.predict(condition)

		return joint_likelihood / (condition_likelihood + 0.0001)