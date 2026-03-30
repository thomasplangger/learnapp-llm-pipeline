import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME   = os.getenv("DB_NAME")
if not MONGO_URL or not DB_NAME:
    raise RuntimeError("MONGO_URL and DB_NAME must be set in .env")

_client = AsyncIOMotorClient(MONGO_URL)
db      = _client[DB_NAME]
fs      = AsyncIOMotorGridFSBucket(db, bucket_name="pdfs")

users_collection     = db.users
courses_collection   = db.courses
lessons_collection   = db.lessons
progress_collection  = db.progress
pdfs_metadata_coll   = db.pdfs_metadata
pdf_texts_coll       = db.pdf_texts
chunks_collection    = db.chunks
