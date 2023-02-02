import sys
sys.path.append('/Users/joseph/Airbnb/modelling-airbnbs-property-listing-dataset-/modelling-airbnbs-property-listing-dataset-')
from tabular_data import *
from prepare_image_data import *

import unittest
import pandas as pd
import numpy as np

class TestRemoveRowsWithMissingRatings(unittest.TestCase):
    def test_remove_rows_with_missing_ratings(self):
        df = pd.read_csv(r'airbnb-property-listings/tabular_data/listing.csv')
        actual_result = remove_rows_with_missing_ratings(df)
        expected_result = pd.read_csv(r'airbnb-property-listings/tabular_data/listing.csv')
        expected_result.drop(columns=['Unnamed: 19'], inplace=True)
        expected_result.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'], inplace=True)
        self.assertTrue(actual_result.equals(expected_result))
        
if __name__ == '__main__':
    unittest.main()




