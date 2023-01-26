import boto3
import os

s3 = boto3.client('s3')
bucket_name = 'joseph-airbnb'
prefix = 'images/'

def get_subfolders(bucket_name, prefix):
    subfolders = []
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Delimiter='/', Prefix=prefix):
        for prefix in result.get('CommonPrefixes', []):
            subfolders.append(prefix.get('Prefix'))
    return subfolders

def download_subfolders_and_images(bucket_name, prefix):
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

download_subfolders_and_images(bucket_name, 'images/')
