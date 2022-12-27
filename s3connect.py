# this script is used to connect to s3 and create a bucket and upload files to the bucket

import logging
import boto3
from botocore.exceptions import ClientError


s3 = boto3.resource(
    service_name='s3',
    region_name='us-west-2',
    aws_access_key_id='AKIAQ5LUBW6SHBHWPDPC',
    aws_secret_access_key='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'
)

#create a new bucket

response = s3.create_bucket(
    Bucket='agingcompaniespapers',   
    CreateBucketConfiguration={
        'LocationConstraint': 'us-west-2',
    },
)

# Print out bucket names

for bucket in s3.buckets.all():
    print(bucket.name)
    
    
# add a files to the bucket
    
s3.Bucket('agingcompaniespapers').upload_file(Filename='post2018_agingcomapnies_papers.csv', Key='papers.csv')
s3.Bucket('agingcompaniespapers').upload_file(Filename='aging_companies/Aging Companies - Companies.csv', Key='companies.csv')