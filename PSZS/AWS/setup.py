from argparse import Namespace
from typing import Optional, Tuple
import warnings
import json
import os
from mypy_boto3_ec2 import EC2Client
from mypy_boto3_s3 import S3Client

from PSZS.AWS.client import initialize_client
from PSZS.AWS.ec2 import get_ebs_volume_path, set_instance_id_env

CONFIG_S3_BUCKET = 's3-bucket'
CONFIG_DATASET_VOLUME_ID = 'dataset-volume-id'

def setup_aws(args: Namespace) -> Tuple[bool, Optional[EC2Client], Optional[S3Client]]:
    """Checks whether AWS is used and sets up the EC2 and S3 clients as well as the updated dataset root path.
    If AWS is not used, the function does nothing. Depending on provided values in the arguments,
    try to load the volume ID and S3 bucket from the config file. 
    Ensures that if either a S3 bucket is provided or local storage is enabled.
    If the dataset root is already an existing directory (absolute path) no modifications are performed,
    otherwise the dataset root is prefixed with the EBS volume path.
    
    Fields loaded from args:
    - aws: Whether AWS is used
    - store_local: Whether to store locally
    - aws_config: Path to the AWS config file
    - dataset_volume_id: Volume ID for the dataset
    - s3_bucket: S3 bucket for the dataset 
        
    Args:
        args (Namespace): Argument namespace object

    Returns:
        Tuple[bool, Optional[EC2Client], Optional[S3Client]]: Whether AWS setup was successful, EC2 client and S3 client
    """
    ec2_client = s3_client = None
    use_aws = getattr(args, 'aws', False)
    if use_aws==False:
        # If not using AWS, do nothing
        print("Not using AWS. Skipping setup.")
        return True, ec2_client, s3_client
    store_local = getattr(args, 'store_local', False)
    aws_config = getattr(args, 'aws_config', None)
    volume_id = getattr(args, 'dataset_volume_id', None)
    s3_bucket = getattr(args, 's3_bucket', None)
    
    # Load the AWS config file if provided and updated not provided values
    if aws_config is not None:
        assert os.path.exists(aws_config), f"AWS config file {aws_config} does not exist."
        print(f"Using AWS config file: {aws_config}")
        with open(aws_config, 'r') as f:
            aws_config_dat: dict = json.load(f)
        if volume_id is None:
            print("Loading volume ID from config.")
            volume_id = aws_config_dat.get(CONFIG_DATASET_VOLUME_ID, None)
        if s3_bucket is None:
            print("Loading S3 Bucket from config.")
            s3_bucket = aws_config_dat.get(CONFIG_S3_BUCKET, None)
    
    # If we can not store locally make sure we have an S3 bucket
    if s3_bucket is None and store_local==False:
        warnings.warn("No S3 bucket provided but no local storage. Either provide bucket, config or set --store-local.")
        return False, ec2_client, s3_client
    
    ec2_client = initialize_client('ec2')
    s3_client = initialize_client('s3')
    if ec2_client is None:
        warnings.warn("EC2 client could not be initialized. Cannot setup AWS.")
        return False, ec2_client, s3_client
    if s3_client is None:
        warnings.warn("S3 client could not be initialized. Cannot setup AWS.")
        return False, ec2_client, s3_client
    
    # Set the EC2_INSTANCE_ID environment variable
    set_instance_id_env()
    if volume_id is not None:
        # If dataset root exists (i.e. was a absolute path) do not update with volume path
        if os.path.exists(args.root) and os.path.isdir(args.root):
            print(f"Dataset root {args.root} exists. Volume ID will be ignored.")
        else:
            # Get the EBS volume path
            ds_volume_path = get_ebs_volume_path(volume_id=volume_id, ec2_client=ec2_client)
            # Update the dataset root
            args.root = os.path.join(ds_volume_path, args.root)
            print(f"Dataset root updated to {args.root}")
    else:
        print(f"No volume ID for dataset provided. Use default root {args.root}")
    return True, ec2_client, s3_client