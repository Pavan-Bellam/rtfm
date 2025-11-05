from langchain_openai import OpenAIEmbeddings
from app.core.settings import settings

#configuratio for chunking model
chunk_config_model = {
    "proivder": "openai",
    "model": "gpt-5-nano",
}

#embedding dimmensions
embed_dim = 1536 
#initialize models that will be used through out the application

openai_embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    api_key=settings.OPENAI_API_KEY,
    dimensions=embed_dim
)