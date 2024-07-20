import os
import shutil
from typing import Optional
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from PSZS.AWS.client import initialize_client

def move_to_s3(local_path: str, bucket_name: str, s3_path: str, s3_client: Optional[S3Client] = None) -> None:
    """
    Move a file or folder from a local path (e.g. the local EBS volume) to an S3 bucket. If no S3 client is provided,
    a new one will be initialized with the default region and credentials using `initialize_client('s3')`.
    After the file or folder is moved, it will be deleted from the local path.
    
    Args:
        local_path (str): The local file or folder path
        bucket_name (str): The name of the S3 bucket
        s3_path (str): The destination path in the S3 bucket
        s3_client (Optional[S3Client], optional): An S3 client object. If None, a new client will be initialized. Defaults to None.
    """
    if s3_client is None:
        print("No S3 client provided. Trying to initializing a new one.")
        s3_client = initialize_client('s3')
        if s3_client is None:
            print("S3 client could not be initialized. Cannot upload to S3 instance.")
            return
    
    try:
        if os.path.isfile(local_path):
            s3_client.upload_file(local_path, bucket_name, s3_path)
            os.remove(local_path)
            print(f"Moved {local_path} to s3://{bucket_name}/{s3_path}")
        elif os.path.isdir(local_path):
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_file_path = os.path.join(s3_path, relative_path)
                    s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
                    os.remove(local_file_path)
                    print(f"Moved {local_file_path} to s3://{bucket_name}/{s3_file_path}")
            shutil.rmtree(local_path)
            print(f"Moved directory {local_path} to s3://{bucket_name}/{s3_path}")
        else:
            print(f"Error: {local_path} is not a valid file or directory")
    except ClientError as e:
        print(f"Error moving to S3: {e}")

def download_from_s3(local_path: str, bucket_name: str, s3_path: str, s3_client: Optional[S3Client] = None) -> None:
    """
    Download a file or folder from an S3 bucket to a local path (e.g. the local EBS volume). If no S3 client is provided,
    a new one will be initialized with the default region and credentials using `initialize_client('s3')`.
    
    Args:
        local_path (str): The local file or folder path
        bucket_name (str): The name of the S3 bucket
        s3_path (str): The destination path in the S3 bucket
        s3_client (Optional[S3Client], optional): An S3 client object. If None, a new client will be initialized. Defaults to None.
    """
    if s3_client is None:
        print("No S3 client provided. Trying to initializing a new one.")
        s3_client = initialize_client('s3')
        if s3_client is None:
            print("S3 client could not be initialized. Cannot download files from S3 instance.")
            return
    
    try:
        # Check if s3_path is a file or a prefix (folder)
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_path)
        
        if 'Contents' in response:
            if len(response['Contents']) == 1 and response['Contents'][0]['Key'] == s3_path:
                # It's a file
                s3_client.download_file(bucket_name, s3_path, local_path)
                print(f"Downloaded s3://{bucket_name}/{s3_path} to {local_path}")
            else:
                # It's a folder
                if not os.path.exists(local_path):
                    os.makedirs(local_path)
                for obj in response['Contents']:
                    file_path = obj['Key']
                    if not file_path.endswith('/'):  # Skip folders
                        local_file_path = os.path.join(local_path, os.path.relpath(file_path, s3_path))
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        s3_client.download_file(bucket_name, file_path, local_file_path)
                        print(f"Downloaded s3://{bucket_name}/{file_path} to {local_file_path}")
        else:
            print(f"Error: s3://{bucket_name}/{s3_path} does not exist")
    except ClientError as e:
        print(f"Error downloading from S3: {e}")