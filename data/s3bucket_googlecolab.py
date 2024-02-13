# -*- coding: utf-8 -*-
"""S3Bucket_GoogleColab

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NAGh1jHanyZL-hJGQuwTqafTfj4KIgO0
"""

# Import Libraries
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir=None):
    """
    Sets up S3 client with given credentials and downloads a specified folder from S3 to a local directory.

    Parameters:
    aws_access_key_id (str): AWS access key ID.
    aws_secret_access_key (str): AWS secret access key.
    bucket_name (str): Name of the S3 bucket.
    s3_folder (str): Folder in the S3 bucket to download.
    local_dir (str, optional): Local directory to download the files to. Defaults to the S3 folder name.
    """

    # Create an S3 client
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    except PartialCredentialsError:
        print("Your AWS credentials are incomplete.")
        return
    except NoCredentialsError:
        print("Your AWS credentials were not found.")
        return
    # Input of function is the bucket name (which is always deeplearningcvfood),
    # the S3 folder which can either be: test, train, validation
    # and a default local directory
    def download_s3_folder(bucket_name, s3_folder, local_dir=None):
        if local_dir is None:
            local_dir = s3_folder
        # AWS S3 paginators are used to retrieve the objects in the bucket in multiple pages
        paginator = s3_client.get_paginator('list_objects_v2')
        try:
            # Iterate over every object on current page
            for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
                for obj in page.get('Contents', []):
                    # Captures the key associate with each image (file)
                    file_key = obj['Key']
                    if file_key.endswith('/'):
                        continue  # skip directories
                    local_file_path = os.path.join(local_dir, os.path.basename(file_key))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    # Actual downloading of S3 object to local file path
                    s3_client.download_file(bucket_name, file_key, local_file_path)
                    print(f"Downloaded {file_key} to {local_file_path}")
        except NoCredentialsError:
            print("Invalid AWS credentials")
        except s3_client.exceptions.NoSuchBucket:
            print("The bucket does not exist or you have no access.")
        except Exception as e:
            print(e)

    # Call the download function
    download_s3_folder(bucket_name, s3_folder, local_dir)

# Usage of function
#aws_access_key_id = 'AKIA6GBMB6KOCGR7U6EQ'
#aws_secret_access_key = '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'
#bucket_name = 'vegetablepictures'
# The different choices for the s3_folder are: test, train, validation
#s3_folder = 'validation'
#local_dir = '/content/drive/My Drive/Project_1/S3_Data/validation/'
#setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir)