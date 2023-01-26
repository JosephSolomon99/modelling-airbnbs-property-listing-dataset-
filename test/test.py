import sys
sys.path.append('/Users/joseph/Airbnb/modelling-airbnbs-property-listing-dataset-/modelling-airbnbs-property-listing-dataset-')
from tabular_data import *
import unittest
import pandas as pd

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.read_csv(r'/Users/joseph/Airbnb/modelling-airbnbs-property-listing-dataset-/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv')
        

    def test_remove_rows_with_missing_ratings(self):
        actual_output = remove_rows_with_missing_ratings(pd.read_csv(r'/Users/joseph/Airbnb/modelling-airbnbs-property-listing-dataset-/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv'))
        self.dataframe.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        expected_output = self.dataframe
        self.assertEqual(actual_output, expected_output)

unittest.main(argv=[''], verbosity=2, exit=False)



