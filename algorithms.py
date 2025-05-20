 ## Algorithms Module
# This module contains the algorithms used in the project.
import numpy
from helper_functions import * 
import itertools
from collections import defaultdict, Counter
import cv2

"""
An implementation of the algorithms described in the paper:
"Star-Tracker Algorithm for Smartphones and Commercial Micro-Drones" by Revital Marbel,Boaz Ben-Moshe and Roi Yozevitch (2020). https://www.mdpi.com/1424-8220/20/4/1106#
Programmers: Yoni Baruch, Yevgeny Ivanov and Daniel Isakov
Date: 01-05-2025
"""

# --- Algorithm 1 Implementation ---

# ! This algorithm is extremely slow, on my M1 Macbook, it iterates over ~100,000 stars a second. There are 10000~ stars in the BSC, and the algorithm's complexity is O(n^3).
# ! Ultimately, it would take this algorithm 10000^3/100000 seconds to run, which is approximately 112.2 days. To test this function, you must know manually the stars in the image, and then run the function with a smaller catalog of stars (100-300) while insuring the original 3 stars are in the catalog subset.
def stars_identification_bf(detected_stars: List[dict],star_catalog: BSC,camera_scaling_factor:float=1) -> Tuple[dict, dict, dict] | None:
    """
    Implements Algorithm 1: Stars identification BF algorithm (Section 2.2).

    Picks one random triplet of pixels from the frame and compares its pixel distances against the angular distances of all possible catalog star triplets to find the catalog triplet yielding the minimum RMS error (using Eq. 1).

    Args:
        detected_stars: A list of detected star pixels (x, y) in the frame, each a dictionary with keys 'x' and 'y'.
        star_catalog: A catalog of bright stars (BSC).
        camera_scaling_factor: A scaling factor to convert pixel distances to angular distances as mentioned in Equation 3. This is not mentioned in the paper, but it is a mathematical necessity to convert the pixel distances to angular distances. The scaling factor is the ratio of the pixel distance to the angular distance, which is a constant for a given camera and setup. Without, the algorithm would not work, and would never converge to a feasible solution.
    Returns:
        The BSC star triplet (s_i, s_j, s_t) that yielded the minimum
        RMS error when compared to the chosen pixel triplet, or None if
        insufficient pixels/stars are available.
        
    Example 1: empty picture without any visible stars via "empty_image.png", scaling factor = 1/16.30.
    >>> stars_identification_bf(detect_stars("test_image.png"),get_star_catalog(),1/16.30)
    None
    
    Example 2: 3 visible stars in the frame via 'test_image.png', scaling factor = 1/16.30, stars are Zaniah, Porrima and Auva.
    >>> stars_identification_bf(detect_stars("test_image.png"),get_star_catalog(),1/16.30)
    ({'B': 'η', 'N': 'Zaniah', 'C': 'Vir', 'Dec': -0.666944, 'F': '15', 'HR': 4689, 'K': '9500', 'RA': 184.976667, 'V': '3.89'}, {'B': 'γ', 'N': 'Porrima', 'C': 'Vir', 'Dec': -1.449444, 'F': '29', 'HR': 4825, 'K': '7500', 'RA': 190.415, 'V': '3.65'}, {'B': 'δ', 'N': 'Auva', 'C': 'Vir', 'Dec': 3.3975, 'F': '43', 'HR': 4910, 'K': '3050', 'RA': 193.900833, 'V': '3.38'})
    """
    if len(detected_stars) < 3:
        print("Error: Need at least 3 stars in the catalog.")
        return None

    # 1. Pick *a* triplet of stars <p1, p2, p3> from Frame
    p1 = detected_stars[0]
    p2 = detected_stars[1]
    p3 = detected_stars[2]

    pixel_triplet = ((p1["x"],p1["y"]),(p2["x"],p2["y"]), (p3["x"],p3["y"]))

    # 2. Calculate sorted distances for the chosen pixel triplet
    dp1 = calculate_pixel_distance(pixel_triplet[0][0],pixel_triplet[0][1],pixel_triplet[1][0],pixel_triplet[1][1]) * camera_scaling_factor
    dp2 = calculate_pixel_distance(pixel_triplet[0][0],pixel_triplet[0][1],pixel_triplet[2][0],pixel_triplet[2][1]) * camera_scaling_factor
    dp3 = calculate_pixel_distance(pixel_triplet[1][0],pixel_triplet[1][1],pixel_triplet[2][0],pixel_triplet[2][1]) * camera_scaling_factor
    sorted_pixel_distances = sorted([dp1, dp2, dp3])

    min_rms = float('inf')
    best_catalog_triplet = None
    # 3. Iterate 'for every 3 stars <si, sj, st> in BSC'
    for catalog_triplet in itertools.combinations(star_catalog, 3):
        s_i, s_j, s_t = catalog_triplet

        # 4. Calculate sorted angular distances for the catalog triplet
        dsi = calculate_angular_distance(s_i['RA'],s_i['Dec'],s_j['RA'],s_j['Dec'])
        dsj = calculate_angular_distance(s_i['RA'],s_i['Dec'],s_t['RA'],s_t['Dec'])
        dst = calculate_angular_distance(s_j['RA'],s_j['Dec'],s_t['RA'],s_t['Dec'])
        sorted_catalog_distances = sorted([dsi, dsj, dst])

        # 5. Calculate RMS using Equation 1
        try:
            current_rms = calculate_rms_error_eq1(
                pixel_distances=sorted_pixel_distances,
                angular_distances=sorted_catalog_distances,
            )
        except ValueError as e:
            print(f"Warning: RMS calculation error for catalog triplet {s_i[0],s_j[0],s_t[0]}: {e}")
            continue # Skip this triplet if distances don't match

        # 6. Update minimum RMS and best matching catalog triplet
        if current_rms < min_rms:
            min_rms = current_rms
            best_catalog_triplet = catalog_triplet
                
    # 7. Return the catalog triplet corresponding to the minimum RMS
    return best_catalog_triplet

# --- Algorithm 2 Implementation ---

def stars_identification(
    detected_stars: List[dict[str, float]], # Original parameter name from your snippet
    spht: dict, 
    al_parameter: float, 
    camera_scaling_factor: float
) -> List[dict[str, any]]: # Output: List of dicts, each with 'coords', 'spht_value', 'confidence'
    """
    Implements Algorithm 2 logic leading to Algorithm 4.
    Uses original list indices of detected_stars as temporary internal IDs.
    Calls setConfidence (Algorithm 4) function.
    Formats the output as requested: (x,y), spht_value, confidence for each star.

    Args:
        detected_stars: List of detected star dictionaries {'x': ..., 'y': ...}.
        spht: The Star Pattern Hash Table.
        al_parameter: Accuracy Level.
        camera_scaling_factor: Scaling factor.
    Returns:
        A list of dictionaries. Each dictionary represents an identified star and contains:
        - 'coords': (x, y) tuple of the original detected star.
        - 'spht_value': The identified catalog star label (string).
        - 'confidence': The confidence score (integer count).
        Stars for which no identification could be made might be omitted or included
        with spht_value=None and confidence=0.
    """
    
    sm_table_for_individual_pixels_by_index = defaultdict(list) # Keys will be indices

    num_detected_stars = len(detected_stars) # Use the correct parameter name

    for index_triplet in itertools.combinations(range(num_detected_stars), 3):
        frame_pixel_triplet_objects = tuple(detected_stars[i] for i in index_triplet) # Use correct param name
        key = create_spht_key(frame_pixel_triplet_objects, al_parameter, camera_scaling_factor)
        matching_catalog_star_triplets = spht.get(key, [])

        if matching_catalog_star_triplets:
            for catalog_star_id_triplet in matching_catalog_star_triplets:
                for i in range(3): 
                    frame_pixel_original_index = index_triplet[i]
                    catalog_star_id = catalog_star_id_triplet[i]
                    sm_table_for_individual_pixels_by_index[frame_pixel_original_index].append(catalog_star_id)
    
    # Call Algorithm 4 (setConfidence)
    confidence_results_by_index = setConfidence(sm_table_for_individual_pixels_by_index) 
    
    # --- Format the output as requested ---
    final_output_list = []
    for original_index, (catalog_label, confidence_val) in confidence_results_by_index.items():
        # Get the original star's coordinates
        original_star_dict = detected_stars[original_index] # Use correct param name
        coordinates = (original_star_dict['x'], original_star_dict['y'])
        
        formatted_entry = {
            "coords": coordinates,
            "spht_value": catalog_label, # This is the best_catalog_label_for_pixel
            "confidence": confidence_val
        }
        final_output_list.append(formatted_entry)
                    
    return final_output_list



# --- Algorithm 3 Implementation ---

def validation_algorithm_orientation(detected_stars: List[dict], orientation_matrix: numpy.ndarray, bsc_catalog: BSC) -> float:
    """
    Implements Algorithm 3: Validation Algorithm for the Reported Orientation.
    
    Algorithm 3 has 2 targets: (i) validates the reported orientation and (ii) improves the accuracy of the RTA orientation result. In order to have a validorientation (T0), at least two stars from the frame need to be matched to corresponding stars from the BSC.

    Args:
        detected_stars: A list of detected star pixels (x, y) in the frame, each a dictionary with keys 'x' and 'y'.
        orientation_matrix: The orientation matrix of the camera. (3x3 numpy array)
        bsc_catalog: The Bright Star Catalog.

    Returns:
        The estimated orientation error (weighted RMS of angular distances),
        or float('inf') if no valid pairs are found or other error.
        
    Example:
    >>> # Setup test data
    >>> import numpy as np
    >>> detected_stars = [{'x': 205, 'y': 30},{'x': 135, 'y': 88},{'x': 46, 'y': 48}]
    >>> # Create a mock identity orientation matrix
    >>> # Create a small mock catalog with just two stars
    >>> mock_catalog = [
    ...     {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
    ...     {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
    ...     {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
    ... ]
    >>> orientation_matrix = calculate_orientation_matrix(detected_stars, mock_catalog)
    >>> # When validation is perfect, error should be very small
    >>> error = validation_algorithm_orientation(detected_stars, orientation_matrix, mock_catalog)
    >>> error < 0.01
    True
    """
    # calculate center of frame image
    img_cv2 = cv2.imread(IMAGE_FILE)
    center_x, center_y, _ = img_cv2.shape
    center_x /= 2
    center_y /= 2

    # #define set ST0=S' - stars S transformed by T0
    # temproray average focal lenght of COTS smartphone camera lenses.
    focal_length = 2700
    # stars_transformed = detect_stars
    detected_stars_coord_vector = []
    for i in range(len(detected_stars)):
        dx = (detected_stars[i].get('x') - center_x) / focal_length
        dy = (detected_stars[i].get('y') - center_y) / focal_length
        norm = sqrt(dx**2 + dy**2 + 1)
        # 3rd dimension is color channel
        coord_vector = [dx/norm, dy/norm, 1/norm]
        detected_stars_coord_vector.append(coord_vector)

    detected_stars_rotated = []
    for i in range(len(detected_stars_coord_vector)):
        rotated_v = detected_stars_coord_vector[i] @ orientation_matrix
        detected_stars_rotated.append(rotated_v)

    # for each star s' in S' *search nearest neighbor* b' from BSC
    # create L a tuple of all such subsets that will be like <s',b'>
    nearest_neighbor_pairs = nearest_neighbor_srch(detected_stars_rotated)

    # based on an "angular error" remove all pairs that exceed this error from L
    for pair in nearest_neighbor_pairs:
        pass
    
    # define ErrorEstimation to be weight root mean squar over 3D distances between pairs in L

    # return ErrorEstimation
    return 0

# --- Algorithm 4 Implementation ---
def setConfidence(sm_table_individual_pixels: dict[int, List[str]]) -> dict:
    """
    Implements Algorithm 4: Best match confidence algorithm.
    Determines the best catalog star label for each INDIVIDUAL frame pixel star
    and its confidence (frequency count).

    Args:
        sm_table_individual_pixels (defaultdict(list) or dict): 
            Maps individual frame pixel IDs (here, original list indices) 
            to a list of candidate catalog star IDs.
            Example: {0: ['sA', 'sD', 'sA'], 1: ['sB', 'sE'], ...} 
                     (where 0, 1 are indices)
    
    Returns:
        Dictionary mapping each individual frame pixel ID (index) 
        to (best_catalog_star_label, confidence_score)
    """
    pixel_final_labels_with_confidence = {}

    for frame_pixel_id, candidate_catalog_labels_for_pixel in sm_table_individual_pixels.items():
        if not candidate_catalog_labels_for_pixel:
            pixel_final_labels_with_confidence[frame_pixel_id] = (None, 0) # No candidates
            continue

        label_counts = Counter(candidate_catalog_labels_for_pixel)
        
        if not label_counts: # Should be caught by the check above, but good for safety
             pixel_final_labels_with_confidence[frame_pixel_id] = (None, 0)
             continue
             
        best_catalog_label_for_pixel, max_count = label_counts.most_common(1)[0]
        confidence_score = max_count # This is the confidence as per Algorithm 4
        pixel_final_labels_with_confidence[frame_pixel_id] = (best_catalog_label_for_pixel, confidence_score)

    return pixel_final_labels_with_confidence

IMAGE_FILE = 'test_image_2.png'
if __name__ == "__main__":
    detected_stars = detect_stars("test_image_2.png")
    star_catalog = [
        {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah', 'B': 'η', 'C': 'Vir', 'F': '15', 'K': '9500', 'V': '3.89'},
        {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima', 'B': 'γ', 'C': 'Vir', 'F': '29', 'K': '7500', 'V': '3.65'},
        {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva', 'B': 'δ', 'C': 'Vir', 'F': '43', 'K': '3050', 'V': '3.38'}
    ]
    camera_scaling_factor = 1/16.30
    al_parameter = 0.1
    spht = {}

    # For each triplet in detected_stars, create a key and put it in spht
    for triplet_indices in itertools.combinations(range(len(detected_stars)), 3):
        triplet = tuple(detected_stars[i] for i in triplet_indices)
        key = create_spht_key(triplet, al_parameter, camera_scaling_factor)
        # Only create a key if the triplet matches the true stars (by coordinates)
        star_coords = [
            {'x': 46, 'y': 48},
            {'x': 205, 'y': 32},
            {'x': 135, 'y': 88}
        ]
        triplet_coords = [{ 'x': t['x'], 'y': t['y'] } for t in triplet]
        if sorted(triplet_coords, key=lambda d: (d['x'], d['y'])) == sorted(star_coords, key=lambda d: (d['x'], d['y'])):
            spht[key] = [
                (star_catalog[0]['HR'], star_catalog[1]['HR'], star_catalog[2]['HR'])
            ]
    # print(spht)
    # Call the stars_identification function
    result = stars_identification(detected_stars, spht, al_parameter, camera_scaling_factor)
    print(result)
