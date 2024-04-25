from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import getpass
import os



from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_openai.embeddings import OpenAIEmbeddings

# from langchain_openai import ChatOpenAI



from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

PINECONE_API_KEY = getpass.getpass("PINEAPI_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

pc = Pinecone()

loader = PyPDFLoader("https://www.warta.pl/documents/UFK/Warunki_ubezpieczen/WN2/WN2_-_OWU_WARTA_NIERUCHOMOSCI.pdf", extract_images=True)
docs = loader.load()


#separators=["§","\n\n", "\n", " ", ""]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=10, separators=["\n§"])
docs = text_splitter.split_documents(docs)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "proj2"
pc.delete_index(index_name)
pc.create_index(
        index_name,
        dimension=1536,
        metric='euclidean',
        spec=spec
    )

embeddings = OpenAIEmbeddings()
index_name = "proj2"
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name, namespace="ns1")