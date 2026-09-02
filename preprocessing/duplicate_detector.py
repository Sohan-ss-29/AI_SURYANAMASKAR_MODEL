"""
==========================================================
Duplicate Detector
----------------------------------------------------------
Detects near-duplicate images using Perceptual Hash (pHash).

Author : Sohan Suhas
Project : AI-Based Surya Namaskar Pose Detection
==========================================================
"""

from PIL import Image
import imagehash


class DuplicateDetector:

    def __init__(self, hash_size=16, threshold=5):
        """
        hash_size:
            Larger hash = more accurate

        threshold:
            Smaller threshold = stricter duplicate detection
        """

        self.hash_size = hash_size
        self.threshold = threshold

    def calculate_hash(self, image_path):
        """
        Calculate perceptual hash of an image.
        """

        image = Image.open(image_path)

        return imagehash.phash(
            image,
            hash_size=self.hash_size
        )

    def is_duplicate(self, image1_path, image2_path):
        """
        Compare two images.

        Returns
        -------
        duplicate : bool
        distance  : int
        """

        hash1 = self.calculate_hash(image1_path)
        hash2 = self.calculate_hash(image2_path)

        distance = hash1 - hash2

        return distance <= self.threshold, distance