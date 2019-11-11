import SimpleITK as sitk
import os
from random import shuffle
import numpy as np

class NIIdataloader(self, positiveDir, negativeDir, shuffle):
	def __init__(self):
		self.positiveDir = positiveDir
		self.negativeDir = negativeDir
		self.directories = [self.positiveDir, self.negativeDir]
		self.trainingData = []
		self.isShuffle = shuffle
	def label(self):
		for i in self.directories:
			for j in os.listdir(i):
				self.path = os.path.join(i, j)
				self.array = sitk.GetArrayFromImage(sitk.ReadImage(path))
				if i == self.positiveDir:
					self.label = 1
				else:
					self.label = 0
				self.trainingData.append([np.array(self.array), np.array(label))
		if self.shuffle == True: 
			self.trainingData = random.shuffle(self.trainingData)
		return self.trainingData

					
