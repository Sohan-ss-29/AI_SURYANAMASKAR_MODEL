"""
==========================================================
Blur Detector
----------------------------------------------------------
Detects blurry images using Variance of Laplacian.

Author : Sohan Suhas
Project : AI-Based Surya Namaskar Pose Detection
==========================================================
"""

import cv2


class BlurDetector:
    """
    Detect blurry images using Variance of Laplacian.
    """

    def __init__(self, threshold=120):
        self.threshold = threshold

    def blur_score(self, image_path):
        """
        Returns the blur score of an image.
        Higher score = Sharper image.
        Lower score = Blurry image.
        """

        image = cv2.imread(image_path)

        if image is None:
            return 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        return score

    def is_blurry(self, image_path):
        """
        Returns:
            True  -> Image is blurry
            False -> Image is sharp
        """

        score = self.blur_score(image_path)

        return score < self.threshold, score