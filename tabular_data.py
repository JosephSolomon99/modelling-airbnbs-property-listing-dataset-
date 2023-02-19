import pandas as pd
import numpy as np

def remove_rows_with_missing_ratings(dataframe: pd.DataFrame):
    dataframe.drop(columns=['Unnamed: 19'], inplace=True)
    dataframe.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'], inplace=True)
    return dataframe
    
def combine_description_strings(dataframe: pd.DataFrame, column_name: str):
    dataframe[column_name].replace("\n"," ",regex=True,inplace=True)
    dataframe[column_name].replace(r'[^A-Za-z0-9\s]+', '',regex=True,inplace=True)
    dataframe[column_name].replace("About this space","",regex=True,inplace=True)
    dataframe[column_name] = dataframe[column_name].str.strip()
    dataframe[column_name].replace(r'\s+', ' ',regex=True,inplace=True)
    dataframe[column_name] = dataframe[column_name].str.lower()
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

def load_airbnb(filepath, label):
    data = pd.read_csv(filepath)
    features = data.drop(columns=label)
    features = features.select_dtypes(include=[np.number])
    labels = data[label]
    return features, labels


def main():
    path = r'airbnb-property-listings/listing.csv'
    raw_data = pd.read_csv(path)
    processed_data = clean_tabular_data(raw_data)
    processed_data.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv', index=False, header=True, encoding='utf-8')  

if __name__ == "__main__":
     main()


