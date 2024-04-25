import getpass
import os

PINECONE_API_KEY = getpass.getpass("PINEAPI_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY