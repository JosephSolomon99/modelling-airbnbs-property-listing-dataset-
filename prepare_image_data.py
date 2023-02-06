import pandas as pd
import boto3
import os
from PIL import Image

def download_subfolders_and_images(bucket_name, prefix):
    s3 = boto3.client('s3')
    def get_subfolders(bucket_name, prefix):
        subfolders = []
        paginator = s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=bucket_name, Delimiter='/', Prefix=prefix):
            for prefix in result.get('CommonPrefixes', []):
                subfolders.append(prefix.get('Prefix'))
        return subfolders
    # create a folder if not exists
    if not os.path.exists("images"):
        os.mkdir("images")   
    subfolders = get_subfolders(bucket_name, prefix)
    for subfolder in subfolders:
        # create a subfolder if not exists
        subfolder_name = subfolder.replace(prefix, '')
        subfolder_path = f'images/{subfolder_name}'
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        for obj in s3.list_objects(Bucket=bucket_name, Prefix=subfolder)['Contents']:
            if obj['Key'].endswith('/'): # to ignore the subfolder
                continue
            try:
                s3.download_file(bucket_name, obj['Key'], f'{subfolder_path}/{obj["Key"].split("/")[-1]}')
                print(f'Successfully downloaded {obj["Key"]}')
            except Exception as e:
                print(f'Error downloading {obj["Key"]}: {e}')

def resize_images(folder_path, new_folder_path):
    smallest_height = float("inf")
    # Find the smallest height among all images
    for subdir,dirs,files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    if img.mode != "RGB":
                        continue
                    width, height = img.size
                    # Update the smallest height
                    smallest_height = min(smallest_height, height)
            except OSError:
                continue              
    # Resize all images
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    if img.mode != "RGB":
                        continue
                    width, height = img.size
                    aspect_ratio = width / height
                    # Calculate the new width based on the aspect ratio and new height
                    new_width = int(smallest_height * aspect_ratio)
                    img = img.resize((new_width, smallest_height), Image.ANTIALIAS)
                    new_filepath = os.path.join(new_folder_path, file)
                    img.save(new_filepath)
            except OSError:
                continue

def main():
    download_subfolders_and_images('joseph-airbnb', 'images/')
    resize_images('images', 'airbnb-property-listings/tabular_data/processed_images')
    
if __name__ == "__main__":
    main()

