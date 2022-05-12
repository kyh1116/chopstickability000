import copy
import csv
import itertools
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

# basically take camera input and convert it into a cv object
# later to be processed by gen()
from script.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


class VideoCamera(object):
	def __init__(self):
		self.video = cv.VideoCapture(1)
		self.video.set(cv.CAP_PROP_FRAME_WIDTH, 960)
		self.video.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

		self.keypoint_classifier = KeyPointClassifier()

		self.use_brect = True

		# Read labels ###########################################################
		with open('script/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
			self.keypoint_classifier_labels = csv.reader(f)
			self.keypoint_classifier_labels = [
				row[0] for row in self.keypoint_classifier_labels
			]

		self.mode = 0

		self.validate = []
		self.invalidate_count = 0
		self.begin = time.time()

		self.passing = False

		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=True,
			max_num_hands=1,
			min_detection_confidence=0.8
		)
		
	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		if success:
			# call the detection here
			image = self.hand_video(success, image)

		return image

	def hand_video(self, flag, frame):
		# For static images:
		# parameters for the detector

		# flip it along y axis
		image = cv.flip(frame, 1)
		debug_image = copy.deepcopy(image)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		# color format conversion
		image.flags.writeable = False
		results = self.hands.process(image)  # 손인식 결과의 객체를 얻어오는 함수
		image.flags.writeable = True

		if results.multi_hand_landmarks is not None:
			# draw result landmarks
			for hand_landmarks in results.multi_hand_landmarks:


				# Bounding box calculation
				brect = self.calc_bounding_rect(debug_image, hand_landmarks)
				# Landmark calculation
				landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

				# Conversion to relative coordinates / normalized coordinates
				pre_processed_landmark_list = self.pre_process_landmark(
					landmark_list)

				# Hand sign classification
				hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

				self.validate.append(hand_sign_id)  # append hand_result index
				if hand_sign_id != self.mode:
					self.invalidate_count += 1

				# Drawing part
				debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
				debug_image = self.draw_hand_classification(debug_image, self.keypoint_classifier_labels[hand_sign_id])

				passed = self.time_checker()

				# 한번 패스되면 패스로 영구적 인식
				if passed == True:
					self.passing = True

		else:
			self.begin = time.time()
			self.passing = False


		debug_image = self.draw_info(debug_image)
		return debug_image

	def calc_bounding_rect(self, image, landmarks):
		image_width, image_height = image.shape[1], image.shape[0]

		landmark_array = np.empty((0, 2), int)

		for _, landmark in enumerate(landmarks.landmark):
			landmark_x = min(int(landmark.x * image_width), image_width - 1)
			landmark_y = min(int(landmark.y * image_height), image_height - 1)

			landmark_point = [np.array((landmark_x, landmark_y))]

			landmark_array = np.append(landmark_array, landmark_point, axis=0)

		x, y, w, h = cv.boundingRect(landmark_array)

		return [x, y, x + w, y + h]

	def calc_landmark_list(self, image, landmarks):
		image_width, image_height = image.shape[1], image.shape[0]

		landmark_point = []

		# Keypoint
		for _, landmark in enumerate(landmarks.landmark):
			landmark_x = min(int(landmark.x * image_width), image_width - 1)
			landmark_y = min(int(landmark.y * image_height), image_height - 1)
			# landmark_z = landmark.z

			landmark_point.append([landmark_x, landmark_y])

		return landmark_point

	def pre_process_landmark(self, landmark_list):
		temp_landmark_list = copy.deepcopy(landmark_list)

		# Convert to relative coordinates
		base_x, base_y = 0, 0
		for index, landmark_point in enumerate(temp_landmark_list):
			if index == 0:
				base_x, base_y = landmark_point[0], landmark_point[1]

			temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
			temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

		# Convert to a one-dimensional list
		temp_landmark_list = list(
			itertools.chain.from_iterable(temp_landmark_list))

		# Normalization
		max_value = max(list(map(abs, temp_landmark_list)))

		def normalize_(n):
			return n / max_value

		temp_landmark_list = list(map(normalize_, temp_landmark_list))

		return temp_landmark_list

	def time_checker(self):
		now = time.time()

		##    print("begin: " , int(begin) , "now: " , int(now))
		if self.validate is None:
			self.invalidate_count = 0
			self.begin = now
			return False

		if now - self.begin >= 6:
			##print("성공")
			self.invalidate_count = 0
			return True
		elif now - self.begin >= 1.5:
			# print(float(invalidate_count) / float(len(validate)))
			if float(self.invalidate_count) / float(len(self.validate)) < 0.2:
				return False
			else:  # (float)invalidate_count/(float)len(validate) >= 0.2:
				self.validate.clear()
				self.begin = now
				self.invalidate_count = 0
				return False
		else:  # now - begin < 1:
			return False

	def draw_bounding_rect(self, use_brect, image, brect):
		if use_brect:
			# Outer rectangle
			cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
						 (0, 0, 0), 1)

		return image

	def select_mode(self, key, mode):
		if key == 48:  # 0
			mode = 0

		elif key == 49:  # 1
			mode = 1

		elif key == 50:  # 2
			mode = 2

		elif key == 51:  # 3
			mode = 3

		return mode

	def draw_hand_classification(self, image, hand_sign_id):
		cv.putText(image, "YOUR STEP: " + hand_sign_id, (10, 120),
				   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
				   cv.LINE_AA)

		return image

	def draw_info(self, image):
		mode_string = ''

		if self.mode == 0:
			mode_string = 'STEP 1'
		elif self.mode == 1:
			mode_string = 'STEP 2'
		elif self.mode == 2:
			mode_string = 'STEP 3'
		elif self.mode == 3:
			mode_string = 'STEP 4'

		cv.putText(image, "MODE:" + mode_string, (10, 90),
				   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
				   cv.LINE_AA)

		if self.passing == True:
			is_passed = "TRUE"
		else:
			is_passed = "FALSE"

		cv.putText(image, "PASSED: " + is_passed, (10, 150),
				   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
				   cv.LINE_AA)

		return image













# generator that saves the video captured if flag is set
def gen(camera, flag):
	if flag == True:
		# time information
		time_now = time.localtime()
		current_time = time.strftime("%H:%M:%S", time_now)
		# default format
		fourcc = cv.VideoWriter_fourcc(*'XVID')
		# output which is a cv writer, given the name and format, and resolution
		out = cv.VideoWriter('output_' + str(current_time) + '.avi',fourcc, 20.0, (640,480))

		while True:
			# cv object to jpg
			ret, jpeg = cv.imencode('.jpg', camera.get_frame())
			# jpg to bytes
			frame =  jpeg.tobytes()
			# generator yielding the bytes
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

			cv_frame = camera.get_frame()
			out.write(cv_frame)
	
	else:
		while True:
			ret, jpeg = cv.imencode('.jpg', camera.get_frame())
			frame =  jpeg.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
