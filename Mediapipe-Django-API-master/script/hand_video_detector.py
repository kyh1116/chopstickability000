import copy
import csv
import itertools
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

from script.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# this method is used to deduce the hand_video method
# plz refer to the hand_video method accordingly
# def hand_image():
#     # For static images:
#     hands = mp_hands.Hands(
#         static_image_mode=True,
#         max_num_hands=1,
#         min_detection_confidence=0.8)
#
#     # feed a video:
#     videoFile = "test_vid.mp4"
#     cap = cv.VideoCapture(videoFile)
#     flag, frame = cap.read()
#
#     # while cap.isOpened():
#     while flag:
#         image = cv.flip(frame, 1)
#         frame_ID = cap.get(1)
#         results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
#         print('Handedness:', results.multi_handedness)
#         if not results.multi_hand_landmarks:
#             continue
#         image_hight, image_width, _ = image.shape
#         annotated_image = image.copy()
#         for hand_landmarks in results.multi_hand_landmarks:
#             print('hand_landmarks:', hand_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
#             )
#             mp_drawing.draw_landmarks(
#                 annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         cv.imwrite(
#             '/tmp/annotated_image_' + str(frame_ID) + '.png', cv.flip(annotated_image, 1))
#         flag, frame = cap.read()
#     hands.close()


def hand_video(flag, frame):
    # For static images:
    # parameters for the detector
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.8)


    # flip it along y axis
    image = cv.flip(frame, 1)
    debug_image = copy.deepcopy(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # color format conversion
    image.flags.writeable = False
    results = hands.process(image)  # 손인식 결과의 객체를 얻어오는 함수
    image.flags.writeable = True

    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        hands.close()
        return frame
    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()
    # draw result landmarks
    for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # flip it back and return
    return cv.flip(annotated_image, 1)

# save the video if user chooese so
def vid_save():
    cap = cv.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
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

    print("begin: ", int(begin), "now: ", int(now))
    if validate is None:
        return now, False

    if now - begin >= 6:
        print("성공")
        return begin, True
    elif now - begin >= 2:
        if float(invalidate_count) / float(len(validate)) >= 0.8:
            return begin, False
        else:  # (float)invalidate_count/(float)len(validate) < 0.8:
            validate.clear()
            return now, False
    else:  # now - begin < 2:
        return begin, False


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def select_mode(key, mode):
    if key == 48:  # 0
        mode = 0

    elif key == 49:  # 1
        mode = 1

    elif key == 50:  # 2
        mode = 2

    elif key == 51:  # 3
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
