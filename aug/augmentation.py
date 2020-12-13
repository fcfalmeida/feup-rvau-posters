import cv2 as cv
import numpy as np
import time
import pickle
from tkinter import Tk
import aug.utils as utils
from aug.options import Options
from prep.database import Database
from prep.camera_calibration import CameraCalibration

import math
#from scipy import ndimage


class Augmentation:

    MIN_GOOD_MATCHES = 30
    RATIO_THRESH = 0.75  # Lowe's Ratio
    CUBE_SIZE = 200
    CUBE_Z_OFFSET = 50
    SPACE_KEY = 32

    def __init__(self):
        self.root = None
        self.db = None
        self.camera_params = CameraCalibration().calibration_params

    def start(self):
        self.root = Tk()
        self.root.withdraw()
        self.db = Database()

        options = Options()

        films = self.db.get_films_with_images()

        detector = options.algorithm
        matcher = cv.DescriptorMatcher_create(
            cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

        utils.tutorial_print(
            f"Starting augmentation using the following parameters:")
        utils.tutorial_print(detector)
        utils.tutorial_print(
            f"Minimum Good Matches: {Augmentation.MIN_GOOD_MATCHES}")
        utils.tutorial_print(
            f"Lowe's Ratio Threshold: {Augmentation.RATIO_THRESH}")

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        if utils.is_tutorial():
            self._tutorial_mode_run(cap, films, detector, matcher)
            return

        while True:
            # Capture frame-by-frame
            success, frame = cap.read()
            # if frame is read correctly ret is True
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            for film, poster_img in films:
                poster_img_gray = cv.cvtColor(poster_img, cv.COLOR_BGR2GRAY)

                # -- Step 1: Detect the keypoints using ORB Detector, compute the descriptors
                utils.tutorial_print(
                    f"Computing keypoints and descriptors for {film.title}...")
                keypoints_obj = film.keypoints
                descriptors_obj = film.descriptors

                keypoints_scene, descriptors_scene = detector.detect_and_compute(gray_frame)

                # Check if any descriptors can be found in the scene
                if descriptors_scene is None:
                    continue

                knn_matches = detector.get_matches(descriptors_obj, descriptors_scene)

                # -- Filter matches using the Lowe's ratio test
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < Augmentation.RATIO_THRESH * n.distance:
                        good_matches.append(m)

                # Try to localize the object only if matches are above a certain value
                if len(good_matches) >= Augmentation.MIN_GOOD_MATCHES:
                    utils.tutorial_print(
                        f"Detected {film.title}'s poster with {len(good_matches)} good matches")
                    # -- Draw matches
                    img_matches = np.empty(
                        (max(poster_img.shape[0], frame.shape[0]), poster_img.shape[1]+frame.shape[1], 3), dtype=np.uint8)

                    # -- Localize the object
                    obj = np.empty((len(good_matches), 2), dtype=np.float32)
                    scene = np.empty((len(good_matches), 2), dtype=np.float32)
                    for i in range(len(good_matches)):
                        # -- Get the keypoints from the good matches
                        obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
                        obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
                        scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
                        scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

                    H, _ = cv.findHomography(obj, scene, cv.RANSAC)
                    # -- Get the corners from the image_1 ( the object to be "detected" )
                    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
                    obj_corners[0, 0, 0] = 0
                    obj_corners[0, 0, 1] = 0
                    obj_corners[1, 0, 0] = poster_img.shape[1]
                    obj_corners[1, 0, 1] = 0
                    obj_corners[2, 0, 0] = poster_img.shape[1]
                    obj_corners[2, 0, 1] = poster_img.shape[0]
                    obj_corners[3, 0, 0] = 0
                    obj_corners[3, 0, 1] = poster_img.shape[0]

                    scene_corners = cv.perspectiveTransform(obj_corners, H)

                    # Top left corner of the poster
                    top_left_corner = (
                        scene_corners[0, 0, 0], scene_corners[0, 0, 1])

                    # Display the name of the film on the poster's top left corner

                    b_channel, g_channel, r_channel = cv.split(frame)

                    # creating a dummy alpha channel image.
                    alpha_channel = np.ones(
                        b_channel.shape, dtype=b_channel.dtype) * 50

                    frame = cv.merge(
                        (b_channel, g_channel, r_channel, alpha_channel))

                    ret, rvecs, tvecs = cv.solvePnP(
                        self._to_3d_points(obj_corners), scene_corners, self.camera_params.mtx, self.camera_params.dist)
                    empty_img = np.zeros(
                        (frame.shape[0], frame.shape[1], 4), dtype="uint8")

                    cv.putText(
                        empty_img, film.title, top_left_corner, cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0, 255), 1, 1)

                    (h, w) = empty_img.shape[:2]
                    M, _ = cv.Rodrigues(rvecs)
                    M = np.delete(M, 2, 0)  # remove z
                    M[0][0] = 1
                    M[1][1] = 1
                    #euler_angles = self._rotationMatrixToEulerAngles(M)

                    # angle = math.sqrt(
                    #           math.degrees(euler_angles[0] ** 2) + math.degrees(euler_angles[1] ** 2))

                    # print(angle)
                    #M = cv.getRotationMatrix2D((w//2, h//2), -angle ,1.0)
                    #empty_img = ndimage.rotate(empty_img, -angle, reshape=False)
                    empty_img = cv.warpAffine(empty_img, M, (w, h))
                    # print(top_left_corner)

                    # cos = np.abs(M[0, 0])
                    # sin = np.abs(M[0, 1])

                    # # compute the new bounding dimensions of the image
                    # nW = int((h * sin) + (w * cos))
                    # nH = int((h * cos) + (w * sin))

                    # perform the actual rotation and return the image
                    cv.imwrite("./transparent_img.png", empty_img)
                    frame = cv.add(frame, empty_img)

                    frame = self._display_score(
                        obj_corners, scene_corners, frame, film.score)

                    cv.imwrite('./frame.png', frame)
                # else:
                    #utils.tutorial_print(f"Found {len(good_matches)} good matches for {film.title}")

            cv.imshow('Augmentation', frame)

            if cv.waitKey(1) == ord('q'):
                self._stop(cap)
                return

        # When everything done, release the capture
        self._stop(cap)

    def _rotationMatrixToEulerAngles(self, R):

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

        # When everything done, release the capture
    def _stop(self, cap):
        cap.release()
        cv.destroyAllWindows()
        # Prevents freezing when closing the window for some reason
        cv.waitKey(1)

    def _display_score(self, obj_corners, scene_corners, frame, score):
        # Calculate the poster's width by subtracting the x coordinate of the top right and top left corner
        poster_width = obj_corners[1, 0, 0] - obj_corners[0, 0, 0]
        # Calculate the poster's height by subtracting the y coordinate of the bottom left and top left corner
        poster_height = obj_corners[3, 0, 1] - obj_corners[0, 0, 1]

        size = Augmentation.CUBE_SIZE

        # Offsets to ensure the cubes are displayed on the center of the poster
        x_offset = poster_width / 2 - size / 2
        y_offset = poster_height / 2 - size / 2

        """
        Find the rotation and translation vectors.
        Here we convert the object's corner's coordinates to 3d points and we assume z = 0
        Since solvePnP expects 3d points
        """
        ret, rvecs, tvecs = cv.solvePnP(
            self._to_3d_points(obj_corners), scene_corners, self.camera_params.mtx, self.camera_params.dist)

        return self._draw_cubes(frame, x_offset, y_offset, score, rvecs, tvecs)

    def _draw_cube(self, frame, cubepts):
        cubepts = np.int32(cubepts).reshape(-1, 2)
        # draw bottom layer
        img = cv.drawContours(frame, [cubepts[:4]], -1, (255, 255, 255), 2)
        # draw pillars
        for i, j in zip(range(4), range(4, 8)):
            img = cv.line(img, tuple(cubepts[i]), tuple(
                cubepts[j]), (255, 255, 255), 2)
        # draw top layer
        img = cv.drawContours(img, [cubepts[4:]], -1, (255, 255, 255), 2)
        return img

    def _draw_cubes(self, frame, x_offset, y_offset, score, rvecs, tvecs):
        size = Augmentation.CUBE_SIZE

        for i in range(score):
            # Translate each cube upwards, leaving CUBE_Z_OFFSET distance between them
            z_offset = -(size + Augmentation.CUBE_Z_OFFSET) * (i+1)
            cube = np.float32([[x_offset, y_offset, z_offset], [x_offset, y_offset+size, z_offset], [x_offset+size, y_offset+size, z_offset], [x_offset+size, y_offset, z_offset],
                               [x_offset, y_offset, -size+z_offset], [x_offset, y_offset+size, -size+z_offset], [x_offset+size, y_offset+size, -size+z_offset], [x_offset+size, y_offset, -size+z_offset]])

            # project 3D points of the cube to image plane
            cubepts, jac = cv.projectPoints(
                cube, rvecs, tvecs, self.camera_params.mtx, self.camera_params.dist)

            frame = self._draw_cube(frame, cubepts)

        return frame

    def _to_3d_points(self, points2d):
        """
        Converts a list of 2d points into a list of 3d points by adding a z = 0 coordinate
        """
        points3d = np.empty((4, 1, 3), dtype=np.float32)

        for i in range(len(points2d)):
            point = points2d[i]
            points3d[i] = [point[0, 0], point[0, 1], 0]

        return points3d

    def _tutorial_mode_run(self, cap, films, detector, matcher):
        success = None
        frame = None
        gray_frame = None
        utils.tutorial_print(
            "Press space while showing the poster to continue")
        while cv.waitKey(1) != Augmentation.SPACE_KEY:
            success, frame = cap.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('Augmentation', frame)

        for film, poster_img in films:
            poster_img_gray = cv.cvtColor(poster_img, cv.COLOR_BGR2GRAY)

            # -- Step 1: Detect the keypoints using ORB Detector, compute the descriptors
            utils.tutorial_print(
                f"Computing keypoints and descriptors for {film.title}...")
            keypoints_obj = film.keypoints
            descriptors_obj = film.descriptors

            keypoints_scene, descriptors_scene = detector.detect_and_compute(gray_frame)

            cv.drawKeypoints(frame, keypoints_scene, frame, color=(255, 0, 0))
            utils.tutorial_print("Detected Keypoints\nPress space to continue")
            cv.imshow('Augmentation', frame)
            while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                pass

            # Check if any descriptors can be found in the scene
            if descriptors_scene is None:
                continue

            knn_matches = detector.get_matches(descriptors_obj, descriptors_scene)

            # -- Filter matches using the Lowe's ratio test
            good_matches = []
            for m, n in knn_matches:
                if m.distance < Augmentation.RATIO_THRESH * n.distance:
                    good_matches.append(m)

            # Try to localize the object only if matches are above a certain value
            if len(good_matches) >= Augmentation.MIN_GOOD_MATCHES:
                utils.tutorial_print(
                    f"Detected {film.title}'s poster with {len(good_matches)} good matches")
                # -- Draw matches
                img_matches = np.empty(
                    (max(poster_img.shape[0], frame.shape[0]), poster_img.shape[1]+frame.shape[1], 3), dtype=np.uint8)

                cv.drawMatches(poster_img, keypoints_obj, frame, keypoints_scene, good_matches,
                               img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                utils.tutorial_print(
                    f"Matches for the poster {film.title}\nPress space to continue")
                cv.imshow('Good Matches', img_matches)
                while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                    pass

                # -- Localize the object
                obj = np.empty((len(good_matches), 2), dtype=np.float32)
                scene = np.empty((len(good_matches), 2), dtype=np.float32)
                for i in range(len(good_matches)):
                    # -- Get the keypoints from the good matches
                    obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
                    obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
                    scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
                    scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

                H, _ = cv.findHomography(obj, scene, cv.RANSAC)
                # -- Get the corners from the image_1 ( the object to be "detected" )
                obj_corners = np.empty((4, 1, 2), dtype=np.float32)
                obj_corners[0, 0, 0] = 0
                obj_corners[0, 0, 1] = 0
                obj_corners[1, 0, 0] = poster_img.shape[1]
                obj_corners[1, 0, 1] = 0
                obj_corners[2, 0, 0] = poster_img.shape[1]
                obj_corners[2, 0, 1] = poster_img.shape[0]
                obj_corners[3, 0, 0] = 0
                obj_corners[3, 0, 1] = poster_img.shape[0]

                scene_corners = cv.perspectiveTransform(obj_corners, H)
                utils.tutorial_print(
                    f"Found scene corners: {scene_corners}\nPress space to continue")
                cv.imshow('Augmentation', frame)
                while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                    pass

                # Top left corner of the poster
                top_left_corner = (
                    scene_corners[0, 0, 0], scene_corners[0, 0, 1])

                # Display the name of the film on the poster's top left corner

                b_channel, g_channel, r_channel = cv.split(frame)

                # creating a dummy alpha channel image.
                alpha_channel = np.ones(
                    b_channel.shape, dtype=b_channel.dtype) * 50

                frame = cv.merge(
                    (b_channel, g_channel, r_channel, alpha_channel))

                ret, rvecs, tvecs = cv.solvePnP(
                    self._to_3d_points(obj_corners), scene_corners, self.camera_params.mtx, self.camera_params.dist)
                empty_img = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype="uint8")

                cv.putText(
                    empty_img, film.title, top_left_corner, cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0, 255), 1, 1)

                (h, w) = empty_img.shape[:2]
                M, _ = cv.Rodrigues(rvecs)
                M = np.delete(M, 2, 0)  # remove z
                M[0][0] = 1
                M[1][1] = 1
                #euler_angles = self._rotationMatrixToEulerAngles(M)

                # angle = math.sqrt(
                #           math.degrees(euler_angles[0] ** 2) + math.degrees(euler_angles[1] ** 2))

                # print(angle)
                #M = cv.getRotationMatrix2D((w//2, h//2), -angle ,1.0)
                #empty_img = ndimage.rotate(empty_img, -angle, reshape=False)
                empty_img = cv.warpAffine(empty_img, M, (w, h))
                # print(top_left_corner)

                # cos = np.abs(M[0, 0])
                # sin = np.abs(M[0, 1])

                # # compute the new bounding dimensions of the image
                # nW = int((h * sin) + (w * cos))
                # nH = int((h * cos) + (w * sin))

                # perform the actual rotation and return the image
                cv.imwrite("./transparent_img.png", empty_img)
                frame = cv.add(frame, empty_img)

                frame = self._display_score(
                    obj_corners, scene_corners, frame, film.score)

                cv.imwrite('./frame.png', frame)
            else:
                utils.tutorial_print(
                    f"The poster didn't have enough good match with {film.title}\nPress space to continue")
                cv.imshow('Augmentation', frame)
                while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                    pass

        utils.tutorial_print("Press space to finish")
        cv.imshow('Augmentation', frame)
        while(cv.waitKey(1) != Augmentation.SPACE_KEY):
            self._stop(cap)
            return
