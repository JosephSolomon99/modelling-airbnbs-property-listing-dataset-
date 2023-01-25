#%%
import pandas as pd

def remove_rows_with_missing_ratings(dataframe: pd.DataFrame):
    dataframe.drop(columns=['Unnamed: 19'], inplace=True)
    dataframe.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe
    
def combine_description_strings(dataframe: pd.DataFrame, column_name: str):
    # Replacing newline characters
    dataframe[column_name].replace("\n"," ",regex=True,inplace=True)
    # Removing special characters
    dataframe[column_name].replace(r'[^A-Za-z0-9\s]+', '',regex=True,inplace=True)
    # Removing phrase "About this space"
    dataframe[column_name].replace("About this space","",regex=True,inplace=True)
    # Stripping whitespaces
    dataframe[column_name] = dataframe[column_name].str.strip()
    # Removing excessive whitespaces
    dataframe[column_name].replace(r'\s+', ' ',regex=True,inplace=True)
    # Converting text to lower case
    dataframe[column_name] = dataframe[column_name].str.lower()
    # Removing rows with missing values
    dataframe.dropna(subset=[column_name],inplace=True)
    return dataframe  
        
def set_default_feature_value(dataframe: pd.DataFrame):
    columns = ['guests','beds','bathrooms','bedrooms']
    [dataframe[column].fillna(value=1, inplace=True) for column in columns]
    return dataframe

def clean_tabular_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = combine_description_strings(dataframe, 'Description')
    dataframe = set_default_feature_value(dataframe)
    return dataframe

def main():
    path = r'airbnb-property-listings/listing.csv'
    raw_data = pd.read_csv(path)
    processed_data = clean_tabular_data(raw_data)
    processed_data.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv', index=False, header=True, encoding='utf-8')

if __name__ == "__main__":
     main()