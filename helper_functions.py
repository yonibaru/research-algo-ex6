import numpy as np
import math
import json
import cv2
import itertools
from typing import List, Tuple, Any
import random

# Define placeholder types for clarity
Pixel = Tuple[float, float] # (x, y) coordinate
Star = Tuple[Any, float, float] # (ID, RA_degrees, DEC_degrees)
Frame = List[Pixel]
BSC = List[dict]
SPHT = Any

def get_star_catalog(file_path='bsc5-short.json') -> BSC:
    """
    Loads and returns the JSON data from 'bsc5-short.json' in the current directory.
    This is a JSON version of the known Yale Bright Star Catalog (BSC) which contains known bright star entries, each with the following keys:
    - 'Dec': Declination in *degrees*
    - 'HR': Harvard Revised Number, which is a unique identifier for the star and can be used to look up additional information via https://www.astro-logger.com/ui/astronomy/search
    - 'K': Effective Temperature, in Kelvin
    - 'RA': Right Ascension in *degrees*
    - 'V': Visual Magnitude
    """
    def ra_to_deg(ra):
        h, m, s = ra.replace('h', ' ').replace('m', ' ').replace('s', '').split()
        return round(15 * (int(h) + int(m) / 60 + float(s) / 3600), 6)

    def dec_to_deg(dec):
        sign = 1 if dec[0] == '+' else -1
        d, m, s = dec[1:].replace('°', ' ').replace('′', ' ').replace('″', '').split()
        return round(sign * (int(d) + int(m) / 60 + float(s) / 3600), 6)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for star in data:
        star['RA'] = ra_to_deg(star['RA'])
        star['Dec'] = dec_to_deg(star['Dec'])
        star['HR'] = int(star['HR'])

    return data

def calculate_pixel_distance(x1:float,y1:float,x2:float,y2:float) -> float:
    """Calculates Euclidean distance between two pixels."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angular_distance(ra1:float, dec1:float, ra2:float, dec2:float) -> float:
    """
    Calculate angular distance between two stars given their right ascension and declination.
    
    Parameters:
        ra1, dec1: Right ascension and declination of first star in degrees
        ra2, dec2: Right ascension and declination of second star in degrees
        
    Returns:
        float: Angular distance in degrees
    """
    # Convert to cartesian coordinates
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)
    
    # Convert to cartesian coordinates
    x1 = math.cos(dec1_rad) * math.cos(ra1_rad)
    y1 = math.cos(dec1_rad) * math.sin(ra1_rad)
    z1 = math.sin(dec1_rad)
    
    x2 = math.cos(dec2_rad) * math.cos(ra2_rad)
    y2 = math.cos(dec2_rad) * math.sin(ra2_rad)
    z2 = math.sin(dec2_rad)
    
    # Calculate dot product
    dot_product = x1*x2 + y1*y2 + z1*z2
    
    # Clamp to prevent numerical errors
    dot_product = max(min(dot_product, 1.0), -1.0)
    
    # Calculate angular distance in radians, then convert to degrees
    angle_radians = math.acos(dot_product)
    angle_degrees = math.degrees(angle_radians)  # More direct than multiplying by (180/π)
    
    return angle_degrees

def calculate_rms_error_eq1(
    pixel_distances: List[float],
    angular_distances: List[float],
) -> float:
    """
    Calculates the Root Mean Square (RMS) error between scaled angular
    distances and pixel distances, implementing the concept from Eq. 1.
    (Implementation provided previously).
    """
    n_pixel = len(pixel_distances)
    n_angular = len(angular_distances)

    if n_pixel != n_angular:
        raise ValueError("Input distance lists must have the same length for RMS.")
    if n_pixel == 0:
        return 0.0

    n = n_pixel
    sum_squared_error = 0.0
    for i in range(n):
        dist_p_pixels = pixel_distances[i]
        dist_s_angular = angular_distances[i]
        error_sq = (dist_s_angular - dist_p_pixels) ** 2
        sum_squared_error += error_sq

    mean_squared_error = sum_squared_error / n
    rms_error = math.sqrt(mean_squared_error)
    return rms_error

def create_spht_key(pixel_triplet_coords: Tuple[dict, dict, dict], al_parameter: float, camera_scaling_factor: float) -> tuple:
    """
    Calculates pairwise distances, sorts, scales, rounds, and returns an SPHT key.
    Args:
        pixel_triplet_coords: Tuple of three star dicts, e.g., ({'x':x1,'y':y1,'id':'p1'}, ...)
                              It's better if these dicts also have a unique 'id' for hashability.
                              Or pass tuples of (x,y) directly if they are used as keys.
        al_parameter: Accuracy level.
        camera_scaling_factor: Scaling factor.
    Returns:
        A tuple representing the SPHT key.
    """
    # Ensure pixel_triplet_coords provides (x,y) for distance calculation
    # This is a placeholder - actual implementation is complex
    coords = [(s['x'], s['y']) for s in pixel_triplet_coords]
    
    p1, p2, p3 = coords
    d12 = calculate_pixel_distance(p1[0], p1[1], p2[0], p2[1]) / camera_scaling_factor
    d13 = calculate_pixel_distance(p1[0], p1[1], p3[0], p3[1]) / camera_scaling_factor
    d23 = calculate_pixel_distance(p2[0], p2[1], p3[0], p3[1]) / camera_scaling_factor
    
    pixel_distances = sorted((d12, d13, d23))
    
    key = tuple((d / al_parameter) * al_parameter for d in pixel_distances) # This rounding must match SPHT key generation exactly
    rounded_key = tuple(round(d, 0) for d in key) # Round to 6 decimal places
    return tuple(sorted(rounded_key)) # Ensure sorted if SPHT keys are always sorted

def create_spht_key_offline(bsc_triplet_coords: Tuple[dict, dict, dict], al_parameter: float,camera_scaling_factor:float=1.0) -> tuple:
    """
    Calculates pairwise distances, sorts, scales, rounds, and returns an SPHT key.
    Args:
        pixel_triplet_coords: Tuple of three star dicts, e.g., ({'x':x1,'y':y1,'id':'p1'}, ...)
                              It's better if these dicts also have a unique 'id' for hashability.
                              Or pass tuples of (x,y) directly if they are used as keys.
        al_parameter: Accuracy level.
        camera_scaling_factor: Scaling factor.
    Returns:
        A tuple representing the SPHT key.
    """
    # Ensure pixel_triplet_coords provides (x,y) for distance calculation
    # This is a placeholder - actual implementation is complex
    coords = [(s['Dec'], s['RA']) for s in bsc_triplet_coords]
    p1, p2, p3 = coords #index 0 = dec, index 1 = ra
    d12 = calculate_angular_distance(p1[1], p1[0], p2[1], p2[0]) 
    d13 = calculate_angular_distance(p1[1], p1[0], p3[1], p3[0]) 
    d23 = calculate_angular_distance(p2[1], p2[0], p3[1], p3[0]) 
    ang_distances = sorted((d12, d13, d23))
    key = tuple((d / al_parameter) * al_parameter for d in ang_distances) 
    rounded_key = tuple(round(d, 0) for d in key) 
    return tuple(sorted(rounded_key)) # Ensure sorted if SPHT keys are always sorted

def detect_stars(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Preprocessing: slight blur to improve detection
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 255

    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 40

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByColor = True
    params.blobColor = 255  # Detect light blobs (stars)

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_blur)

    stars = []
    for kp in keypoints:
        x, y = kp.pt
        r = kp.size / 2
        brightness = img[int(round(y)), int(round(x))]
        stars.append({"x": int(round(x)), "y": int(round(y)), "r": r, "b": brightness})

    return stars

def visualize_stars(image_path, stars, output_path="stars_detected.png"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    for star in stars:
        center = (star["x"], star["y"])
        radius = int(round(star["r"]))
        cv2.circle(image, center, radius, (0, 255, 0), 1)
        cv2.putText(image, f"({star['x']},{star['y']})", 
                    (star["x"] + 5, star["y"] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if output_path:
        cv2.imwrite(output_path, image)

    return image

def calculate_orientation_matrix(stars, bsc_matches, image_resolution=(250, 134)):
    """
    Calculate the orientation matrix from the detected stars and the star catalog.

    Parameters:
    - detected_stars: List of dictionaries with 'x' and 'y' pixel coordinates of detected stars.
    - bsc_catalog: List of dictionaries with 'RA' and 'Dec' of stars from the catalog.
    - image_resolution: Tuple (width, height) of the image resolution (default is (250, 134)).

    Returns:
    - Orientation matrix (rotation matrix from camera frame to inertial frame)
    """

    def pixel_to_camera_vector(x, y, cx, cy, f=1):
        """Convert pixel (x, y) to a unit vector in the camera frame."""
        vector = np.array([x - cx, y - cy, f])
        return vector / np.linalg.norm(vector)

    def ra_dec_to_inertial_vector(ra_deg, dec_deg):
        """Convert RA/Dec (in degrees) to a 3D unit vector in the inertial frame."""
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        return np.array([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ])

    # Image resolution and principal point (center of the image)
    cx, cy = image_resolution[0] / 2, image_resolution[1] / 2

    # Step 1: Convert the detected star positions to unit vectors in the camera frame
    camera_vectors = np.array([pixel_to_camera_vector(star['x'], star['y'], cx, cy) for star in stars])

    # Step 2: Convert the BSC star catalog (RA/Dec) to unit vectors in the inertial frame
    inertial_vectors = np.array([ra_dec_to_inertial_vector(star['RA'], star['Dec']) for star in bsc_matches])

    # Step 3: Use the Kabsch algorithm to find the optimal rotation matrix
    def kabsch_algorithm(A, B):
        """Find the rotation matrix that minimizes the RMSD between two sets of points A and B."""
        H = np.dot(A.T, B)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        return R

    # Compute the rotation matrix using Kabsch algorithm
    R = kabsch_algorithm(camera_vectors, inertial_vectors)

    return R