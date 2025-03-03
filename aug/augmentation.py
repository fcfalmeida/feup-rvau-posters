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

    RATIO_THRESH = 0.75  # Lowe's Ratio
    CUBE_SIZE = 100
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

        print(f"Starting augmentation using the following parameters:")
        print(detector)
        print(f"Minimum Good Matches: {detector.min_good_matches}")
        print(f"Lowe's Ratio Threshold: {Augmentation.RATIO_THRESH}")
        print("Using solvePnPRansac" if options.use_solvePnpRansac else "Using solvePnP")

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        if utils.is_tutorial():
            self._tutorial_mode_run(cap, films, detector, matcher)
            return

        while True:
            try:
                # Capture frame-by-frame
                success, frame = cap.read()
                # if frame is read correctly success is True
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                # Convert the frame intro grayscale so that it can be processed
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # Detect the keypoints and compute the descriptors
                keypoints_scene, descriptors_scene = detector.detect_and_compute(
                        gray_frame)

                # Check if any descriptors can be found in the scene
                if descriptors_scene is None:
                    continue

                for film, poster_img in films:
                    poster_img_gray = cv.cvtColor(
                        poster_img, cv.COLOR_BGR2GRAY)

                    # Grab the poster's keypoints and descriptors
                    keypoints_obj = film.keypoints
                    descriptors_obj = film.descriptors

                    # Find matches between the descriptors of the poster image and the frame
                    knn_matches = detector.get_matches(
                        descriptors_obj, descriptors_scene)

                    # Filter out matches below the threshold
                    good_matches = self._find_good_matches(knn_matches)

                    # Try to localize the object only if matches are above a certain value
                    if len(good_matches) >= detector.min_good_matches:
                        H, obj_corners = self._find_homography(
                            poster_img, good_matches, keypoints_obj, keypoints_scene)

                        scene_corners = cv.perspectiveTransform(obj_corners, H)

                        frame = self._display_title(
                            frame, scene_corners, film.title)

                        frame = self._display_score(
                            obj_corners, scene_corners, frame, film.score)

                cv.imshow('Augmentation', frame)

                if cv.waitKey(1) == ord('q'):
                    self._stop(cap)
                    return

            except:
                pass

        self._stop(cap)

    def _stop(self, cap):
        cap.release()
        cv.destroyAllWindows()
        # Prevents freezing when closing the window for some reason
        cv.waitKey(1)

    def _find_good_matches(self, matches):
        """ Filter out matches according to RATIO_THRESH

        Parameters
        ----------
        matches: Descriptor matches

        Returns
        -------
        list of float
            The list of matches that pass the ratio test
        """
        good_matches = []
        for m, n in matches:
            if m.distance < Augmentation.RATIO_THRESH * n.distance:
                good_matches.append(m)

        return good_matches

    def _find_homography(self, poster_img, good_matches, keypoints_obj, keypoints_scene):
        """Computes the homography of the poster's image

        Parameters
        ----------
        poster_img : np.array 
            The image file of the poster 
        good_matches : list 
            The list of good matches obtained from _find_good_matches 
        keypoints_obj : np.array  
            The keypoints identified in the poster's image 
        keypoints_scene : np.array 
            The keypoints identified in the frame (scene) 
            
        Returns
        -------
        tuple of np.array 
            a tuple containing the homography matrix and the poster's corners' scene coordinates
        """

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

        return H, obj_corners

    def _display_title(self, frame, scene_corners, title):
        """Displays the title of the movie on the top left corner of the poster in the scene

        Paramaters
        ----------
        frame : np.array
            The currently displayed camera image frame
        scene_corners : np.array
            The coordinates of the poster's corners in the scene
        title : str
            The title of the movie

        Returns
        -------
        np.array
            The frame with the text on the top left corner of the poster
        """
        try:
            # Display the name of the film on the poster's top left corner
            b_channel, g_channel, r_channel = cv.split(frame)

            # creating a dummy alpha channel image.
            alpha_channel = np.ones(
                b_channel.shape, dtype=b_channel.dtype) * 50

            frame = cv.merge(
                (b_channel, g_channel, r_channel, alpha_channel))

            text_img = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype="uint8")

            # Top left corner of the poster
            top_left_corner = (
                scene_corners[0, 0, 0], scene_corners[0, 0, 1])

            cv.putText(
                text_img, title, top_left_corner, cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0, 255), 1, 1)

            frame_scene_points = []
            frame_scene_points.append([
                scene_corners[0, 0, 0], scene_corners[0, 0, 1]])  # top left
            frame_scene_points.append([
                scene_corners[1, 0, 0], scene_corners[1, 0, 1]])  # top right
            frame_scene_points.append([
                scene_corners[2, 0, 0], scene_corners[2, 0, 1]])  # bottom right
            frame_scene_points.append([
                scene_corners[3, 0, 0], scene_corners[3, 0, 1]])  # bottom left
            offset_x = math.sqrt(
                (scene_corners[0, 0, 0] - scene_corners[1, 0, 0]) ** 2 +
                (scene_corners[0, 0, 1] - scene_corners[1, 0, 1]) ** 2)
            text_img_points = np.float32(
                [[scene_corners[0, 0, 0], scene_corners[0, 0, 1]],
                    [scene_corners[0, 0, 0]+offset_x, scene_corners[0, 0, 1]],
                    [scene_corners[0, 0, 0]+offset_x, scene_corners[3, 0, 1]],
                    [scene_corners[0, 0, 0], scene_corners[3, 0, 1]]])
            frame_scene_points = np.float32(frame_scene_points)
            transformation_matrix = cv.getPerspectiveTransform(
                text_img_points, frame_scene_points)
            (h, w) = text_img.shape[:2]
            text_img = cv.warpPerspective(
                text_img, transformation_matrix, (w, h))

            frame = cv.add(frame, text_img)

            return frame
        except ValueError:
            return frame

    def _display_score(self, obj_corners, scene_corners, frame, score):
        """Draws a stack of cubes in the center of the poster according to the movie's score

        Parameters
        ----------
        obj_corners : np.array
            The coordinates of the poster image's corners
        scene_corners : np.array
            The coordinates of the poster's corners in the scene
        frame : np.array
            Currently displayed frame
        score : int
            The movie's score

        Returns
        -------
        np.array
            The frame with a stack of cubes placed in the center of the poster
        """

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
        if (Options().use_solvePnpRansac):
            _, rvecs, tvecs, inliers = cv.solvePnPRansac(
                self._to_3d_points(obj_corners), scene_corners, self.camera_params.mtx, self.camera_params.dist, 
                useExtrinsicGuess=True, iterationsCount=300, reprojectionError=3.0)
                
            if (inliers is None or len(inliers) < 4):
                return frame
        else:
             _, rvecs, tvecs = cv.solvePnP(
                self._to_3d_points(obj_corners), scene_corners, self.camera_params.mtx, self.camera_params.dist)

        return self._draw_cubes(frame, x_offset, y_offset, score, rvecs, tvecs)

    def _draw_cube(self, frame, cubepts):
        """Draws a cube according to the provided vertex coordinates

        Parameters
        ----------
        frame : np.array
            Currently displayed frame
        cubepts : np.array
            The coordinates of the cube's vertices

        Returns
        -------
        np.array
            The frame with a cube drawn in the specified coordinates
        """
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
        """Draws a number of cubes according to the movie's score

        Parameters
        ----------
        frame : np.array
            Currently displayed frame
        x_offset : int
            The x axis offset from which to start drawing the cubes
        y_offset : int
            The y axis offset from which to start drawing the cubes
        score : int
            The movie's score
        rvecs : np.array
            Rotation vectors obtained from solvePnP
        tvecs : np.array
            Translation vectors obtained from solvePnP

        Returns
        -------
        np.array
            The frame with a stack of cubes in the center of the poster
        """
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
        """Converts an array of 2d points into an array of 3d points by adding a z = 0 coordinate

        Parameters
        ----------
        points2d : np.array
            An array of 2D points

        Returns
        -------
        np.array
            An array of 3D points with z = 0
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
        print(
            "Press space while showing the poster to continue")
        while cv.waitKey(1) != Augmentation.SPACE_KEY:
            success, frame = cap.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                return
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('Augmentation', frame)

        for film, poster_img in films:
            poster_img_gray = cv.cvtColor(poster_img, cv.COLOR_BGR2GRAY)

            # -- Step 1: Detect the keypoints using ORB Detector, compute the descriptors
            print(
                f"Computing keypoints and descriptors for {film.title}...")
            keypoints_obj = film.keypoints
            descriptors_obj = film.descriptors

            keypoints_scene, descriptors_scene = detector.detect_and_compute(
                gray_frame)

            cv.drawKeypoints(frame, keypoints_scene,
                             frame, color=(255, 0, 0))
            print("Detected Keypoints\nPress space to continue")
            cv.imshow('Augmentation', frame)
            while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                pass

            # Check if any descriptors can be found in the scene
            if descriptors_scene is None:
                continue

            knn_matches = detector.get_matches(
                descriptors_obj, descriptors_scene)

            good_matches = self._find_good_matches(knn_matches)

            # Try to localize the object only if matches are above a certain value
            if len(good_matches) >= detector.min_good_matches:
                print(
                    f"Detected {film.title}'s poster with {len(good_matches)} good matches")
                # -- Draw matches
                img_matches = np.empty(
                    (max(poster_img.shape[0], frame.shape[0]), poster_img.shape[1]+frame.shape[1], 3), dtype=np.uint8)

                cv.drawMatches(poster_img, keypoints_obj, frame, keypoints_scene, good_matches,
                               img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                print(
                    f"Matches for the poster {film.title}\nPress space to continue")
                cv.imshow('Good Matches', img_matches)

                while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                    pass

                H, obj_corners = self._find_homography(
                    poster_img, good_matches, keypoints_obj, keypoints_scene)

                scene_corners = cv.perspectiveTransform(obj_corners, H)

                print(
                    f"Found scene corners: {scene_corners}\nPress space to continue")

                cv.imshow('Augmentation', frame)
                while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                    pass

                frame = self._display_title(
                    frame, scene_corners, film.title)

                frame = self._display_score(
                    obj_corners, scene_corners, frame, film.score)
                break

            else:
                print(
                    f"The poster didn't have enough good match with {film.title}\nPress space to continue")
                cv.imshow('Augmentation', frame)
                while(cv.waitKey(1) != Augmentation.SPACE_KEY):
                    pass

        print("Press space to finish")
        cv.imshow('Augmentation', frame)

        while(cv.waitKey(1) != Augmentation.SPACE_KEY):
            pass
        self._stop(cap)
        return
