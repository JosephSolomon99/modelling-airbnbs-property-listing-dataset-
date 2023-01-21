import pandas as pd
filepath = r'airbnb-property-listings/tabular_data/listing.csv'


def clean_tabular_data(filepath):
    '''

    '''
    def read_dataset():
        listings = pd.read_csv(filepath)
        return listings

    def remove_rows_with_missing_ratings():
        '''

        '''
        listings = read_dataset()
        listings.drop(columns=['Unnamed: 19'], inplace=True)
        listings.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'], inplace=True)
        listings.reset_index(drop=True, inplace=True)
        return listings
    
    def combine_description_strings():
        '''

        '''
        def remove_special_characters():
            '''

            '''
            listings = remove_rows_with_missing_ratings()
            listings['description'] = listings['description'].apply(remove_special_characters)

        def remove_emoji():
            '''

            '''
            pass
        
        def remove_punctuation():
            '''

            '''
            pass

        def remove_URL():
            pass

        def remove_html():
            pass

        def lower():
            pass
    
    def combine_price_strings():
        '''

        '''
    
    def combine_bedrooms_strings():
        '''

        '''
    
    def combine_bathrooms_strings():
        '''

        '''
    
    def combine_sqft_strings():
        '''

        '''
    
    def combine_sqft_per_unit_strings():
        '''

        '''
    
    def combine_sqft_per_unit_and_bedrooms_strings():
        '''

        '''
    
    def combine_sqft_per_unit_and_bathrooms_strings():

