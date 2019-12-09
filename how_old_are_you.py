import argparse
import glob
import os
import sys

import cv2
import dlib
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_file
import numpy as np
import scipy.io

from wide_resnet import WideResNet


parser = argparse.ArgumentParser()

parser.add_argument(
	'-t', '--train',
	action='store_true',
	help='train network',
)

parser.add_argument(
	'-w', '--weight_file',
	help='specify path to weight file',
)

parser.add_argument(
	'-i', '--images',
	default='images/',
	help='path to your test images directory'
)


def create_database_from_utk_dataset():
	'''
	create database from ./UTKFace directory -- ages from 0 to 100
	(if age is over 100, we will save it as 100)
	'''
	IMAGE_SIZE = 64
	ages = []
	images = []
	for image_path in glob.glob('training/*.jpg'):
		age = image_path.split(os.sep)[-1].split('_')[0]
		out_age = min(int(age), 100)
		ages.append(out_age)
		img = cv2.imread(image_path)
		resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
		images.append(resized)

	db = {
		'image': np.array(images),
		'age': np.array(ages),
		'db': 'utk',
		'img_size': IMAGE_SIZE,
		'min_score': -1,
	}

	scipy.io.savemat('utk.mat', db)


class Schedule:
	'''
	Used in training to adjust learning rate
	'''
	def __init__(self, nb_epochs, initial_lr):
		self.epochs = nb_epochs
		self.initial_lr = initial_lr

	def __call__(self, epoch_idx):
		if epoch_idx < self.epochs * 0.25:
			return self.initial_lr
		elif epoch_idx < self.epochs * 0.50:
			return self.initial_lr * 0.2
		elif epoch_idx < self.epochs * 0.75:
			return self.initial_lr * 0.04
		return self.initial_lr * 0.008


def train_from_db():
	'''
	train the solution gadget: CNN "WideResNet" 
	WideResNet is designed to predict both age and gender; I am omitting
	gender prediction for my project
	'''

	data = scipy.io.loadmat('utk.mat')

	# initialize model for training
	model = WideResNet(data['img_size'][0, 0])()
	model.compile(
		optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
		loss=["categorical_crossentropy", "categorical_crossentropy"],
		metrics=['accuracy'],
	)
	model.summary()

	nb_epochs = 32
	lr = 0.1

	callbacks = [
		LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
		ModelCheckpoint(
			"weights.{epoch:02d}-{val_loss:.2f}.hdf5",
			monitor="val_loss",
			verbose=1,
			save_best_only=True,
			mode="auto",
		)
	]

	X_data = data['image']
	y_data_a = np_utils.to_categorical(data['age'][0], 101)
	y_data_g = np_utils.to_categorical([0] * len(data['age'][0]), 2)	# place holder for unused gender output

	indexes = np.arange(len(X_data))
	np.random.shuffle(indexes)

    # shuffle the dataset
	X_data = X_data[indexes]
	y_data_a = y_data_a[indexes]

    # train on 90%; validate on 10%
	train_num = int(len(X_data) * (1 - 0.1))
	X_train = X_data[:train_num]
	X_test = X_data[train_num:]
	y_train_g = y_data_g[:train_num]
	y_test_g = y_data_g[train_num:]
	y_train_a = y_data_a[:train_num]
	y_test_a = y_data_a[train_num:]

    # training
	hist = model.fit(
		X_train,
		[y_train_g, y_train_a],
		batch_size=32,
		epochs=nb_epochs,
		callbacks=callbacks,
		validation_data=(X_test, [y_test_g, y_test_a]),
	)


def get_weight_file_from_repo():
	print('Getting weight file from repo...')
	weight_file = get_file(
		'weights.32-3.73.hdf5',
		'https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5',
		cache_dir=os.getcwd(),
	)
	return weight_file


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


if __name__ == '__main__':

	args = parser.parse_args()
	arg_dict = vars(args)

	if arg_dict['train']:
		if not os.path.exists('utk.mat'):
			try:
				print('Creating database file from UTK image dataset')
				create_database_from_utk_dataset()
				print('Created database file "utk.mat"')
			except:
				print('cannot find images in ./UTKFace dir')
				sys.exit(0)
		
		print('Training network from image database...')
		try:
			train_from_db()
			print('Training complete!')
		except:
			print('Could not find UTKFace training dataset in "./training" directory')
	
		sys.exit(0)

	weight_file = arg_dict['weight_file'] or get_weight_file_from_repo()
	model = WideResNet(64)()
	model.load_weights(weight_file)

	detector = dlib.get_frontal_face_detector()
	margin = 0.4

	for image_path in glob.glob(arg_dict['images'] + '/*'):
		img = cv2.imread(str(image_path), 1)
		if img is None:
			continue
		h, w, _ = img.shape
		r = 640 / max(w, h)
		resized = cv2.resize(img, (int(w * r), int(h * r)))
		input_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
		img_h, img_w, _ = np.shape(input_img)
		detected = detector(input_img, 1)
		faces = np.empty((len(detected), 64, 64, 3))

		if detected:
			for i, d in enumerate(detected):
            	# rectangle around each face detected in image
				x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
				xw1 = max(int(x1 - margin * w), 0)
				yw1 = max(int(y1 - margin * h), 0)
				xw2 = min(int(x2 + margin * w), img_w - 1)
				yw2 = min(int(y2 + margin * h), img_h - 1)
				cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
				faces[i, :, :, :] = cv2.resize(resized[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))

            # predict age for each face in image
			results = model.predict(faces)
			ages = np.arange(0, 101).reshape(101, 1)
			predicted_ages = results[1].dot(ages).flatten()

            # draw results
			for d in detected:
				label = str(int(predicted_ages[i]))
				draw_label(resized, (d.left(), d.top()), label)

        # show image until waitkey is pushed
		cv2.imshow("result", resized)
		if cv2.waitKey(-1) == 27:  # ESC
			break


