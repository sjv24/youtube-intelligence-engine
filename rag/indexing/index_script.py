import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes import query
from langchain_community.vectorstores import FAISS
from utils.general_functions import readJson, getList
from pipeline.embedding import local_embeddings, getLocalEmbedding

client = weaviate.connect_to_local()
data = readJson("structured_sents.json")
basic_chunks = getList("basic_chunks.csv", "chunks")

# -------- Weaviate --------
def getProperty():
    properties = []
    properties.append(Property(name="Par_ID",
                    data_type=DataType.INT,
                    description='Paragraph id',
                    indexFilterable=True,
                    indexSearchable=True,
                    vectorize_property_name=False,
                    skip_vectorization=True))
    properties.append(Property(name="Title",
                    data_type=DataType.TEXT,
                    description='Section Title',
                    indexFilterable=True,
                    indexSearchable=True,
                    vectorize_property_name=False,
                    skip_vectorization=True))
    properties.append(Property(name="Text",
                    data_type=DataType.TEXT,
                    description='Chunk Text',
                    indexFilterable=True,
                    indexSearchable=True))
    return properties

def createCollection(name):
    properties = getProperty()
    if client.collections.exists(name):
        client.collections.delete(name)
    collection = client.collections.create(
        name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef=128,
                ef_construction=128,
                max_connections=64,
            ),
        ),
        properties=properties,
        inverted_index_config=Configure.inverted_index(
            bm25_k1=1.2,
            bm25_b=0.75,
        )
    )

def loadDataIntoCollection(batch_size, data, collection_name):
    collection = client.collections.use(collection_name)
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for a_chunk in data:
            vector_ = getLocalEmbedding(a_chunk["chunk"])
            object_ = {
                "Par_ID": a_chunk["par_num"] + 1,
                "Title": a_chunk["title"],
                "Text": a_chunk["chunk"]
            }
            batch.add_object(properties=object_, vector=vector_)

# -------- FAISS --------
def createFAISSIndex(chunks):
    faiss_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "vector_stores", "faiss_store")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=local_embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore

collection_name = "Interstellar_Script"
batch_size = 10

createCollection(collection_name)
loadDataIntoCollection(batch_size, data, collection_name)
createFAISSIndex(basic_chunks)

# test
vector_ = getLocalEmbedding("Cooper leaving Murph")
collection = client.collections.use(collection_name)
docs = collection.query.near_vector(
    near_vector=vector_,
    limit=3,
    return_metadata=query.MetadataQuery(certainty=True)
)
print([(obj.properties["text"], obj.metadata.certainty) for obj in docs.objects])

client.close()