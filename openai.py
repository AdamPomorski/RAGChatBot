import getpass
import os

OPENAI_API_KEY = getpass.getpass("OPENAPI_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY