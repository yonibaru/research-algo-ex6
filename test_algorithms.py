import pytest
import random
import itertools
from typing import List, Tuple, Dict, Any
from algorithms import stars_identification_bf
from helper_functions import detect_stars
import numpy as np

# We'll assume these helper functions for the tests
# In a real test scenario, these would be imported from the appropriate module
def calculate_pixel_distance(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def calculate_angular_distance(ra1, dec1, ra2, dec2):
    return abs(ra2-ra1) + abs(dec2-dec1)  # Simplified for testing

def calculate_rms_error_eq1(pixel_distances, angular_distances):
    return sum((p-a)**2 for p, a in zip(pixel_distances, angular_distances))**0.5

class TestStarsIdentificationBF:
    

    # Test case 1: Empty input - no detected stars
    def test_empty_input(self):
        detected_stars = []
        star_catalog = [
            {'RA': 100.0, 'Dec': 20.0, 'HR': 1, 'N': 'Star1'},
            {'RA': 110.0, 'Dec': 25.0, 'HR': 2, 'N': 'Star2'},
            {'RA': 120.0, 'Dec': 30.0, 'HR': 3, 'N': 'Star3'}
        ]
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        assert result is None, "Function should return None for empty input"
    
    # Test case 2: Insufficient stars (less than 3)
    def test_insufficient_stars(self):
        detected_stars = [
            {'x': 100, 'y': 200},
            {'x': 150, 'y': 250}
        ]
        star_catalog = [
            {'RA': 100.0, 'Dec': 20.0, 'HR': 1, 'N': 'Star1'},
            {'RA': 110.0, 'Dec': 25.0, 'HR': 2, 'N': 'Star2'},
            {'RA': 120.0, 'Dec': 30.0, 'HR': 3, 'N': 'Star3'}
        ]
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        assert result is None, "Function should return None for insufficient stars"
    
    # Test case 3: Perfect match - known stars
    def test_perfect_match(self):
        detected_stars = detect_stars("test_image.png") 
        
        
        star_catalog = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah', 'B': 'η', 'C': 'Vir', 'F': '15', 'K': '9500', 'V': '3.89'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima', 'B': 'γ', 'C': 'Vir', 'F': '29', 'K': '7500', 'V': '3.65'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva', 'B': 'δ', 'C': 'Vir', 'F': '43', 'K': '3050', 'V': '3.38'}
        ]
        
        expected_result = tuple(star_catalog)
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        
        assert result == expected_result, f"Expected {expected_result}, but got {result}"
    
    # Test case 4: No matching stars in catalog
    def test_no_matching_stars(self):
        detected_stars = [
            {'x': 100, 'y': 100},
            {'x': 200, 'y': 200},
            {'x': 300, 'y': 300}
        ]
        
        # Create a star catalog with very different angular distances
        star_catalog = [
            {'RA': 10.0, 'Dec': 10.0, 'HR': 1, 'N': 'Star1'},
            {'RA': 20.0, 'Dec': 20.0, 'HR': 2, 'N': 'Star2'},
            {'RA': 30.0, 'Dec': 30.0, 'HR': 3, 'N': 'Star3'}
        ]
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        
        # The function should still return a triplet even with high RMS error
        assert result is not None, "Function should return a triplet even with high RMS error"
        assert len(result) == 3, "Function should return exactly 3 stars"
    
    
    # Test case 6: Invalid scaling factor
    def test_invalid_scaling_factor(self):
        detected_stars = detect_stars("test_image.png")
        
        star_catalog = []
        star_catalog[0] = {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'}
        star_catalog[1] = {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'}
        star_catalog[2] = {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        
        # Test with zero scaling factor (should raise an error)
        with pytest.raises(ValueError):
            stars_identification_bf(detected_stars, star_catalog, 0)
        
        # Test with negative scaling factor (should raise an error)
        with pytest.raises(ValueError):
            stars_identification_bf(detected_stars, star_catalog, -1/16.30)
    
    # Test case 7: Large and random catalog test
    def test_large_catalog(self):
        detected_stars = detect_stars("test_image.png")
        
        # Create a large catalog of 100 stars (a lot of outliers), randomly generated
        star_catalog = [
            {'RA': random.uniform(0, 360), 'Dec': random.uniform(-90, 90), 
             'HR': i, 'N': f'Star{i}'} for i in range(1, 100)
        ]
        
        # Add three stars that should match better than others
        star_catalog[0] = {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'}
        star_catalog[1] = {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'}
        star_catalog[2] = {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        
        # The test should verify that a triplet is returned
        assert result is not None, "Function should return a triplet"
        assert len(result) == 3, "Function should return exactly 3 stars"
        
        # For a full catalog, the function would likely timeout, but we can't test that directly with pytest
        # We can assert that the implementation would be very slow
        assert len(list(itertools.combinations(star_catalog, 3))) > 100000000, "Full catalog would have too many combinations"
        

from algorithms import stars_identification_improved
from helper_functions import create_spht_key

# Mock SPHT class for testing purposes
class SPHT:
    def __init__(self, patterns=None):
        self.patterns = patterns or {}
    
    def lookup(self, key):
        return self.patterns.get(key, [])

class TestStarsIdentificationImproved:
    
    # Test case 1: Empty input - no detected stars
    def test_empty_input(self):
        detected_stars = []
        spht = SPHT()
        al_parameter = 0.1
        
        result = stars_identification_improved(detected_stars, spht, al_parameter, 1/16.30)
        assert result == [], "Function should return empty list for empty input"
    
    # Test case 2: Insufficient stars (less than 3)
    def test_insufficient_stars(self):
        detected_stars = [
            {'x': 100, 'y': 200},
            {'x': 150, 'y': 250}
        ]
        spht = SPHT()
        al_parameter = 0.1
        
        result = stars_identification_improved(detected_stars, spht, al_parameter, 1/16.30)
        assert result == [], "Function should return empty list for insufficient stars"
    
    # Test case 3: Perfect match with pre-populated SPHT
    def test_perfect_match(self):
        # Three detected stars in the frame
        detected_stars = [
            {'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48}
        ]
        
        # Create catalog stars that should match
        catalog_stars = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        
        # Create a pre-populated SPHT with a pattern that matches our detected stars
        # The key would normally be generated from the sorted distances
        # For testing purposes, we'll use a simple key
        al_parameter = 0.2

        pattern_key = create_spht_key(catalog_stars[0], catalog_stars[1], catalog_stars[2], al_parameter)
        spht = SPHT({pattern_key: [catalog_stars]})
    
        result = stars_identification_improved(detected_stars, spht, al_parameter, 1/16.30)
        
        # Expected result: Match between detected stars and catalog stars
        expected_result = [[Tuple(detected_stars),Tuple(catalog_stars)]]
        
        assert result == expected_result, f"Expected {expected_result}, but got {result}"
    
    
    # Test case 4: Multiple matches in SPHT
    def test_multiple_matches(self):
        detected_stars = [
            {'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48}
        ]
        
        # Create multiple catalog star triplets that match the pattern
        catalog_triplet1 = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        
        catalog_triplet2 = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N':'StarOutlier1'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'StarOutlier2'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'StarOutlier3'}
        ]
        
        # SPHT with multiple matching patterns for the same key
        al_parameter = 0.2
        pattern_key1 = create_spht_key(catalog_triplet1[0], catalog_triplet1[1], catalog_triplet1[2], al_parameter)
        pattern_key2 = create_spht_key(catalog_triplet2[0], catalog_triplet2[1], catalog_triplet2[2], al_parameter)
        spht = SPHT({pattern_key1: [catalog_triplet1]},{pattern_key2: [catalog_triplet2]})
        
        result = stars_identification_improved(detected_stars, spht, al_parameter, 1/16.30)
        
        # Expected: Both matches should be returned, with the better match first
        expected_result = [
            (tuple(detected_stars), tuple(catalog_triplet1)),
            (tuple(detected_stars), tuple(catalog_triplet2))
        ]
        
        # Verify that we got both matches (order might vary)
        assert len(result) == 2, "Function should return two matches"
        for match in expected_result:
            assert match in result, f"Expected match {match} not found in results"
    
    
    # Test case 5: Large number of detected stars (hundreds of outliers)
    def test_large_number_of_detected_stars(self):
        # Generate 200 random star positions
        detected_stars = [
            {'x': random.randint(0, 1000), 'y': random.randint(0, 1000)} for _ in range(200)
        ]
        
        real_stars = [
            {'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48}
        ]
        
        detected_stars = detected_stars + real_stars
        
        # Create a simple SPHT
        catalog_triplet = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        
        al_parameter = 0.2
        key = create_spht_key(catalog_triplet[0], catalog_triplet[1], catalog_triplet[2], al_parameter)
        spht = SPHT({key: [catalog_triplet]})
        
        result = stars_identification_improved(detected_stars, spht, al_parameter, 1/16.30)
        
        # The function should handle a large number of stars efficiently
        # and potentially find multiple matches among the combinations
        
        # We can't predict exactly how many matches will be found,
        # but we can verify the structure of the results
        for r in result:
            assert len(r) == 2, "Each result should be a tuple with two elements"
            detected_triplet, catalog_triplet = r
            assert len(detected_triplet) == 3, "Detected triplet should have 3 stars"
            assert len(catalog_triplet) == 3, "Catalog triplet should have 3 stars"
            
            # Check that the detected stars are from our input list
            for star in detected_triplet:
                assert star in detected_stars, "Returned star should be from the input list"
                
from algorithms import validation_algorithm_orientation
from helper_functions import calculate_orientation_matrix

class TestValidationAlgorithmOrientation:

    def test_no_detected_stars(self):
        detected_stars = []
        orientation_matrix = np.eye(3)
        bsc_catalog = []
        result = validation_algorithm_orientation(detected_stars, orientation_matrix, bsc_catalog)
        assert result == None

    def test_insufficient_matches(self):
        detected_stars = [{'x': 200, 'y': 300}]
        bsc_catalog = [{'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'}]
        orientation_matrix = calculate_orientation_matrix(detected_stars, bsc_catalog)
        result = validation_algorithm_orientation(detected_stars, orientation_matrix, bsc_catalog)
        assert result == None

    def test_valid_minimum_case(self):
        detected_stars = [            
            {'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48}]
        bsc_catalog = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        orientation_matrix = calculate_orientation_matrix(detected_stars, bsc_catalog)
        result = validation_algorithm_orientation(detected_stars, orientation_matrix, bsc_catalog)
        assert isinstance(result, float)
        assert result <= 0.5 # good result

    def test_large_random_bsc(self):
        detected_stars = [{'x': random.uniform(0, 1024), 'y': random.uniform(0, 1024)} for _ in range(10)]
        detected_stars += [{'x': 205, 'y': 30},{'x': 135, 'y': 88}, {'x': 46, 'y': 48}]
        orientation_matrix = calculate_orientation_matrix(detected_stars, bsc_catalog)
        bsc_catalog = [
            {
                'RA': random.uniform(0, 360),
                'Dec': random.uniform(-90, 90),
                'HR': i,
                'N': f'Star{i}'
            } for i in range(100)
        ]
        result = validation_algorithm_orientation(detected_stars, orientation_matrix, bsc_catalog)
        assert isinstance(result, float)
        assert result <= 0.5 # good result
        
    def test_large_random_bsc_2(self):
        detected_stars = [{'x': random.uniform(0, 1024), 'y': random.uniform(0, 1024)} for _ in range(10)]
        detected_stars += [{'x': 205, 'y': 30},{'x': 135, 'y': 88}, {'x': 46, 'y': 48}]
        orientation_matrix = np.eye(3) #orientation matrix is just an identity matrix, wrong matrix in this case.
        bsc_catalog = [
            {
                'RA': random.uniform(0, 360),
                'Dec': random.uniform(-90, 90),
                'HR': i,
                'N': f'Star{i}'
            } for i in range(100)
        ]
        result = validation_algorithm_orientation(detected_stars, orientation_matrix, bsc_catalog)
        assert isinstance(result, float)
        assert result >= 5 # bad result

from algorithms import best_match_confidence_algorithm

class TestBestMatchConfidenceAlgorithm:

    def test_empty_inputs(self):
        detected_stars = []
        ism_data: List[Tuple[Tuple[Dict,Dict,Dict], Tuple[Dict,Dict,Dict]]] = []
        result = best_match_confidence_algorithm(detected_stars, ism_data)
        assert result == []

    def test_single_pixel_single_match(self):
        detected_stars = [            
            ({'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48})]
        bsc_catalog = [(
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        )]
        # One triplet containing p1 once, matching to s1
        ism_data = [
            (detected_stars[0], bsc_catalog[0])
        ]
        out = best_match_confidence_algorithm(detected_stars, ism_data)
        # Expect one entry with confidence 1.0
        assert len(out) == 1
        pix, star, conf = out[0]
        assert conf == pytest.approx(1.0)

    def test_multiple_occurrences(self):
       detected_stars = [            
            ({'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48}),(
            {'x': 100, 'y': 100},
            {'x': 200, 'y': 200},
            {'x': 300, 'y': 300})
        ]
       bsc_catalog = [(
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        )]
       ism_data = [
            (detected_stars[0], bsc_catalog[0]),(detected_stars[1], bsc_catalog[0]),
        ]
       out = best_match_confidence_algorithm(detected_stars, ism_data)
       assert len(out) == 6 # 2 triplets * 3 stars each
       

    def test_large_random_data(self):
        detected_stars = [
            {'x': random.randint(0, 1000), 'y': random.randint(0, 1000)} for _ in range(200)
        ]
        real_stars = [
            {'x': 205, 'y': 30},
            {'x': 135, 'y': 88},
            {'x': 46, 'y': 48}
        ]
        detected_stars = detected_stars + real_stars
        # Create a simple SPHT
        catalog_triplet = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        ism_data = [
            (real_stars,catalog_triplet)
        ]
        out = best_match_confidence_algorithm(detected_stars, ism_data)
        assert len(out) == 3 # 1 triplet * 3 stars each