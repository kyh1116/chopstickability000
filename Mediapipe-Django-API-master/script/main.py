#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0) #카메라 관련
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.8)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.8)

    args = parser.parse_args()

    return args


def main():
    count = [0]
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()


    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]


    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)



    #  ########################################################################
    mode = 0 #기본모드
    
    validate = []
    invalidate_count = 0
    begin = time.time()

    passing = False

    
    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)

        mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image) #손인식 결과의 객체를 얻어오는 함수
        image.flags.writeable = True


        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)



                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                validate.append(hand_sign_id) #append hand_result index
                if hand_sign_id != mode:
                    invalidate_count += 1
                    

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_hand_classification(debug_image, keypoint_classifier_labels[hand_sign_id])

                begin, passed = time_checker(begin, validate, invalidate_count)

                #한번 패스되면 패스로 영구적 인식
                if passed == True:
                    passing = True

        else:
            begin = time.time()
            print(int(begin))
            passing = False
            



                
        debug_image = draw_info(debug_image, fps, mode, passing)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()





def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
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



def time_checker(begin, validate, invalidate_count):
    now = time.time()

    print("begin: " , int(begin) , "now: " , int(now))
    if validate is None:
        return now, False
    
    if now - begin >= 6:
        print("성공")
        return begin, True
    elif now - begin >= 2:
        if float(invalidate_count)/float(len(validate)) >= 0.8:
            return begin, False
        else: #(float)invalidate_count/(float)len(validate) < 0.8:
            validate.clear()
            return now, False
    else: #now - begin < 2:
        return begin, False
        
        
        
    
    


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image




def select_mode(key, mode):

    if key == 48:  #0
        mode = 0
        
    elif key == 49: #1
        mode = 1
        
    elif key == 50: #2
        mode = 2

    elif key == 51: #3
        mode = 3
        
    return mode


def draw_hand_classification(image, hand_sign_id):

    cv.putText(image, "YOUR STEP: " + hand_sign_id, (10, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
    
    return image
    

def draw_info(image, fps, mode, passing):

    mode_string = ''
    
    if mode == 1:
        mode_string = 'STEP 1'
    elif mode == 2:
        mode_string = 'STEP 2'
    elif mode == 3:
        mode_string = 'STEP 3'
    elif mode == 4:
        mode_string = 'STEP 4'
    
    cv.putText(image, "MODE:" + mode_string, (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

    if passing == True:
        is_passed = "TRUE"
    else:
        is_passed = "FALSE"

    cv.putText(image, "PASSED: " + is_passed, (10, 150),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
               cv.LINE_AA)

    return image






if __name__ == '__main__':
    main()
