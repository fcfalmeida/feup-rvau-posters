import cv2 as cv
import numpy as np
import time
import pickle
from tkinter import Tk
from prep.database import Database
from prep.camera_calibration import CameraCalibration


class Augmentation:

    MIN_GOOD_MATCHES = 120
    HESSIAN_THRESHOLD = 400
    CUBE_SIZE = 200
    CUBE_Z_OFFSET = 50

    def __init__(self, cbrows=9, cbcols=6):
        self.root = None
        self.db = Database()
        self.camera_params = CameraCalibration().calibration_params

    def start(self):
        self.root = Tk()
        self.root.withdraw()

        films = self.db.get_films_with_images()

        detector = cv.xfeatures2d_SURF.create(Augmentation.HESSIAN_THRESHOLD)
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

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

                # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
                keypoints_obj = film.keypoints
                descriptors_obj = film.descriptors

                keypoints_scene, descriptors_scene = detector.detectAndCompute(
                    gray_frame, None)

                # Check if any descriptors can be found in the scene
                if descriptors_scene is None:
                    continue

                # -- Step 2: Matching descriptor vectors with a FLANN based matcher
                # Since SURF is a floating-point descriptor NORM_L2 is used
                knn_matches = matcher.knnMatch(
                    descriptors_obj, descriptors_scene, 2)

                # -- Filter matches using the Lowe's ratio test
                ratio_thresh = 0.75
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

                print(f"{film.title}:{len(good_matches)}")
                # Try to localize the object only if matches are above a certain value
                if len(good_matches) >= Augmentation.MIN_GOOD_MATCHES:
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

                    cv.putText(
                        frame, film.title, top_left_corner, cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 1)

                    frame = self._display_score(
                        obj_corners, scene_corners, frame, film.score)

            cv.imshow('Augmentation', frame)

            if cv.waitKey(1) == ord('q'):
                self._stop(cap)
                return

        # When everything done, release the capture
        self._stop(cap)

    def _stop(self, cap):
        cap.release()
        cv.destroyAllWindows()
        # Prevents freezing when closing the window for some reason
        cv.waitKey(1)

    def _display_score(self, obj_corners, scene_corners, frame, score):
        poster_width = obj_corners[1, 0, 0] - obj_corners[0, 0, 0]
        poster_height = obj_corners[3, 0, 1] - obj_corners[0, 0, 1]

        size = Augmentation.CUBE_SIZE

        x_offset = poster_width / 2 - size / 2
        y_offset = poster_height / 2 - size / 2

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(
            self._to_3d_points(obj_corners), scene_corners, self.camera_params.mtx, self.camera_params.dist)

        return self._draw_cubes(frame, x_offset, y_offset, score, rvecs, tvecs)

    def _draw_cube(self, frame, cubepts):
        cubepts = np.int32(cubepts).reshape(-1, 2)
        # draw ground floor in green
        img = cv.drawContours(frame, [cubepts[:4]], -1, (255, 255, 255), 2)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv.line(img, tuple(cubepts[i]), tuple(
                cubepts[j]), (255, 255, 255), 2)
        # draw top layer in red color
        img = cv.drawContours(img, [cubepts[4:]], -1, (255, 255, 255), 2)
        return img

    def _draw_cubes(self, frame, x_offset, y_offset, score, rvecs, tvecs):
        size = Augmentation.CUBE_SIZE

        for i in range(score):
            z_offset = -(size + Augmentation.CUBE_Z_OFFSET) * i
            cube = np.float32([[x_offset, y_offset, z_offset], [x_offset, y_offset+size, z_offset], [x_offset+size, y_offset+size, z_offset], [x_offset+size, y_offset, z_offset],
                               [x_offset, y_offset, -size+z_offset], [x_offset, y_offset+size, -size+z_offset], [x_offset+size, y_offset+size, -size+z_offset], [x_offset+size, y_offset, -size+z_offset]])

            # project 3D points of the cube to image plane
            cubepts, jac = cv.projectPoints(
                cube, rvecs, tvecs, self.camera_params.mtx, self.camera_params.dist)

            frame = self._draw_cube(frame, cubepts)

        return frame

    def _to_3d_points(self, points2d):
        points3d = np.empty((4, 1, 3), dtype=np.float32)

        for i in range(len(points2d)):
            point = points2d[i]
            points3d[i] = [point[0, 0], point[0, 1], 0]

        return points3d
