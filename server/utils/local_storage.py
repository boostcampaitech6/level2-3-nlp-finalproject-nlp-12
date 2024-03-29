import boto3
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def connect_s3():
    s3 = boto3.client('s3')
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    logger.info("S3 connected")
    return s3, BUCKET_NAME

def upload_file_to_s3(s3, BUCKET_NAME, file_path, object_path):
    s3.upload_file(file_path, BUCKET_NAME, object_path)

def download_file_from_s3(s3, BUCKET_NAME, object_path, file_path):
    s3.download_file(BUCKET_NAME, object_path, file_path)