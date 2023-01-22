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
    path = r'airbnb-property-listings/tabular_data/listing.csv'
    tabular_data = pd.read_csv(path)
    clean_tabular_data = clean_tabular_data(tabular_data)
    clean_tabular_data.to_csv(os.path.join(path,r'clean_tabular_data.csv', index=False, header=True, encoding='utf-8'))
    return

if __name__ == "__main__":
    
     main()
    

