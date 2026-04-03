import os
import certifi
import pymongo
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("MONGO_DB_URL")
ca = certifi.where()

print(f"Connecting to: {url[:30]}...")  # prints partial URL for safety

try:
    client = pymongo.MongoClient(url, tlsCAFile=ca)
    print(client.list_database_names())
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}")