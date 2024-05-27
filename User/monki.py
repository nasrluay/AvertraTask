import boto3
from datetime import datetime

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

# Specify your bucket name
bucket_name = 'avertra-utility-bot'

# List objects in the specified bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Initialize a variable to keep track of the latest object
latest_object = None

# Iterate through the objects and find the latest one
for obj in response.get('Contents', []):
    if latest_object is None or obj['LastModified'] > latest_object['LastModified']:
        latest_object = obj

if latest_object:
    print(f"Latest object key: {latest_object['Key']}")
    print(f"Last modified: {latest_object['LastModified']}")
else:
    print("Bucket is empty or no objects found.")
