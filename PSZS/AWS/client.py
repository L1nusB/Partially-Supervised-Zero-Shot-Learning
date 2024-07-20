from typing import Optional, Any
import warnings
import boto3
from botocore.exceptions import NoRegionError
from botocore.client import BaseClient
from boto3.session import Session

def initialize_client(service_name: str, region_name: Optional[str]=None, **session_params) -> BaseClient | None:
    """Initialize boto3 client with optional region specification.

    Args:
        region_name (Optional[str], optional): 
            AWS region name (e.g., 'us-west-2'). If None, will try to use the default. Defaults to None.
        session_params: 
            Additional parameters to pass to boto3.Session (e.g. `aws_access_session_id`, 
             `aws_secret_access_key`, `aws_session_token`).

    Returns:
        BaseClient | None: Client object if successful, None otherwise.
    """
    try:
        session: Session = boto3.Session(region_name=region_name, **session_params)
        cli: BaseClient = session.client(service_name)
        return cli
    except NoRegionError:
        warnings.warn("No region specified and no default region found in AWS configuration. "
                      "Please specify a region or configure your AWS CLI with 'aws configure'.")
        return None