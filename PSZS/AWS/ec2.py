from typing import Optional
import os
import requests
from botocore.exceptions import ClientError
from mypy_boto3_ec2 import EC2Client

from PSZS.AWS.client import initialize_client

def get_instance_id() -> str | None:
    """
    Retrieve the EC2 instance ID using IMDSv2 to generate a token and make a request to the metadata service.
    
    Returns:
        str | None: The instance ID or None if not available
    """
    try:
        # First, get a session token
        token_url = "http://169.254.169.254/latest/api/token"
        token_headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
        token_response = requests.put(token_url, headers=token_headers, timeout=2)
        if token_response.status_code != 200:
            print(f"Failed to retrieve IMDSv2 token. Status code: {token_response.status_code}")
            return None

        token = token_response.text

        # Now, use the token to get the instance ID
        instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
        instance_id_headers = {"X-aws-ec2-metadata-token": token}
        response = requests.get(instance_id_url, headers=instance_id_headers, timeout=2)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve instance ID. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error retrieving instance ID from metadata service: {e}")
        return None

def set_instance_id_env(instance_id: Optional[str] = None) -> None:
    """
    Set the EC2_INSTANCE_ID environment variable. If no instance ID is provided, attempt to retrieve it
    using `get_instance_id()`.
    
    Args:
        instance_id (Optional[str], optional): Instance ID specifier. Defaults to None.
    """
    if instance_id is None:
        instance_id = get_instance_id()
    if instance_id:
        os.environ['EC2_INSTANCE_ID'] = instance_id
        print(f"Set EC2_INSTANCE_ID environment variable to {instance_id}")
    else:
        print("Failed to set EC2_INSTANCE_ID environment variable")

def stop_ec2_instance(ec2_client: Optional[EC2Client] = None,
                      instance_id: Optional[str] = None) -> None:
    """
    Stop the EC2 instance of the specified `instance_id`. If no `instance_id` is provided 
    stops the EC2 instance this script is running on. 
    If no EC2 client is provided, try to retrieve it from the environment varialble `EC2_INSTANCE_ID`.
    If this is not set a one will be initialized using `initialize_client('ec2')` 
    with the default region and credentials.
    
    Args:
        ec2_client (Optional[EC2Client], optional): An EC2 client object. If None, a new client will be initialized. Defaults to None.
        instance_id (Optional[str], optional): Instance ID to stop. Defaults to None.
    """
    if ec2_client is None:
        print("No EC2 client provided. Trying to initializing a new one.")
        ec2_client = initialize_client('ec2')
        if ec2_client is None:
            print("EC2 client could not be initialized. Cannot stop EC2 instance.")
            return
    try:
        if instance_id is None:
            instance_id = os.environ.get('EC2_INSTANCE_ID')
            if not instance_id:
                instance_id = get_instance_id()
        
        if not instance_id:
            print("Error: Unable to retrieve instance ID.")
            return
        
        ec2_client.stop_instances(InstanceIds=[instance_id])
        print(f"Stopping EC2 instance {instance_id}")
    except ClientError as e:
        print(f"Error stopping EC2 instance: {e}")
        
def get_ebs_volume_path(volume_id: str, ec2_client: Optional[EC2Client] = None) -> str:
    """
    Get the file path of an EBS volume on the EC2 instance for the given `volume_id`.
    If no EC2 client is provided, try to retrieve it from the environment varialble `EC2_INSTANCE_ID`.
    If this is not set a one will be initialized using `initialize_client('ec2')`.
    
    Args:
        volume_id (str): Volume ID to receive the path for.
        ec2_client (Optional[EC2Client], optional): An EC2 client object. If None, a new client will be initialized. Defaults to None.
        
    Returns:
        str: The path of the EBS volume on the EC2 instance
    """
    if ec2_client is None:
        print("No EC2 client provided. Trying to initializing a new one.")
        ec2_client = initialize_client('ec2')
        if ec2_client is None:
            print("EC2 client could not be initialized. Cannot stop EC2 instance.")
            return
    
    try:
        response = ec2_client.describe_volumes(VolumeIds=[volume_id])
        if response['Volumes']:
            attachments = response['Volumes'][0]['Attachments']
            if attachments:
                device_name = attachments[0]['Device']
                # The actual path might vary depending on the OS and device naming scheme
                return f"/dev/{device_name.split('/')[-1]}"
                # return f"/dev/xvd{device_name[-1]}"
        return None
    except ClientError as e:
        print(f"Error getting EBS volume path: {e}")
        return None