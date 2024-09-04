from langchain_cohere import CohereEmbeddings

from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import os

os.environ['PINECONE_API_KEY'] = 'a89c0db2-bf43-4d8b-ad20-6a78f6a460aa'
os.environ['COHERE_API_KEY'] = 'fnG3bEeE72QgamxF2bXIupoHHwPBttKKHbhZvg09'

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
documents = [item for sublist in docs for item in sublist]

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=2000,
    chunk_overlap=100,

)

chunked_docs = text_splitter.split_documents(documents)

vectordb = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = "workshop"

if index_name not in vectordb.list_indexes().names():
    vectordb.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )


docsearch = PineconeVectorStore.from_documents(chunked_docs, embeddings, index_name=index_name)

print(docsearch.similarity_search("what is agent memory?", k=3))

# Load Pinecone