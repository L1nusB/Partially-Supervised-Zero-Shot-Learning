from typing import Optional
from argparse import Namespace
from mypy_boto3_ec2 import EC2Client
from mypy_boto3_s3 import S3Client

from PSZS.AWS.client import initialize_client
from PSZS.AWS.s3 import move_to_s3
from PSZS.AWS.ec2 import stop_ec2_instance
from PSZS.Utils import is_other_sh_scripts_running

def handle_aws_postprocessing(args: Namespace,
                              s3_client: Optional[S3Client] = None,
                              ec2_client: Optional[EC2Client] = None) -> None:
    """Handle postprocessing for AWS. If AWS is not used, no postprocessing is needed.
    If the logs are not stored locally, upload them to the S3 bucket. Finally, stop the EC2 instance.

    Args:
        args (Namespace): Argument namespace object
        s3_client (Optional[S3Client], optional): S3 client object. If not provided default is constructed. Defaults to None.
        ec2_client (Optional[EC2Client], optional): EC2 client object. If not provided default is constructed. Defaults to None.
    """
    if getattr(args, 'aws', False)==False:
        print("Not using AWS. No AWS postprocessing needed.")
        return
    
    if getattr(args, 'store_local', False) == False:
        if s3_client is None:
            print("No S3 client provided. Trying to initializing a new one.")
            s3_client = initialize_client('s3')
            if s3_client is None:
                print("S3 client could not be initialized. Cannot upload to S3 instance.")
                return
        # After setup it is guaranteed that a S3 bucket is provided when store_local is False
        s3_bucket = getattr(args, 's3_bucket')
        log_root = args.root
        print(f"Uploading logs from {log_root} to s3://{s3_bucket}/{log_root}")
        move_to_s3(local_path=log_root, bucket_name=s3_bucket, s3_path=log_root, s3_client=s3_client)
    
    if getattr(args, "stop_instance", False):
        print("Shutting down EC2 instance.")
        if ec2_client is None:
            print("No EC2 client provided. Trying to initializing a new one.")
            ec2_client = initialize_client('ec2')
            if ec2_client is None:
                print("EC2 client could not be initialized. Cannot stop EC2 instance.")
                return
        if getattr(args, "force_stop", False):
            print("Force stopping EC2 instance.")
            stop_ec2_instance(ec2_client=ec2_client)
        else:
            if is_other_sh_scripts_running():
                print("Other shell scripts are running. Not stopping EC2 instance.")
            else:
                print("No other shell scripts are running. Stopping EC2 instance.")
                stop_ec2_instance(ec2_client=ec2_client)