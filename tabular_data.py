import pandas as pd
import os

def clean_tabular_data():
    '''

    '''
    def remove_rows_with_missing_ratings(dataframe: pd.DataFrame):
        '''
        
        
        '''
        dataframe.drop(columns=['Unnamed: 19'], inplace=True)
        dataframe.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe
    
    def combine_description_strings():
        '''

        '''
        pass

        def remove_special_characters():
            '''

            '''
            pass

        def remove_emoji():
            '''

            '''
            pass
        
        def remove_punctuation():
            '''

            '''
            pass

        def remove_URL():
            '''
            
            '''
            pass

        def remove_html():
            '''
            
            '''
            pass

        def lower():
            '''
            
            '''
            pass
    
    def set_default_feature_value(dataframe: pd.DataFrame):
        '''

        '''
        columns = ['guests','beds','bathrooms','bedrooms']
        [dataframe[column].fillna(value=1, inplace=True) for column in columns]

def main():
    filepath = r'airbnb-property-listings/tabular_data/listing.csv'
    listings = pd.read_csv(filepath)
    clean_tabular_data = clean_tabular_data(listings)
    clean_tabular_data.to_csv('clean_tabular_data.csv', index=False, header=True, encoding='utf-8')
    return

if __name__ == "__main__":
    
     main()
    

