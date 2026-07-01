import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes import query
from utils.general_functions import getCSVPath
from pipeline.embedding import openai_embeddings

client = weaviate.connect_to_local()

# load comments
csv_path = os.path.join(getCSVPath(), "absa_results_full.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} comments")

def getProperty():
    properties = []
    properties.append(Property(name="Comment_ID",
                    data_type=DataType.INT,
                    description='Comment index',
                    indexFilterable=True,
                    indexSearchable=True,
                    vectorize_property_name=False,
                    skip_vectorization=True))
    properties.append(Property(name="Text",
                    data_type=DataType.TEXT,
                    description='Comment text',
                    indexFilterable=True,
                    indexSearchable=True))
    properties.append(Property(name="Sentiment",
                    data_type=DataType.TEXT,
                    description='ABSA sentiment',
                    indexFilterable=True,
                    indexSearchable=True,
                    vectorize_property_name=False,
                    skip_vectorization=True))
    properties.append(Property(name="Aspects",
                    data_type=DataType.TEXT,
                    description='ABSA aspects',
                    indexFilterable=True,
                    indexSearchable=True,
                    vectorize_property_name=False,
                    skip_vectorization=True))
    return properties

def createCollection(name):
    properties = getProperty()
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
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

from tqdm import tqdm

def loadDataIntoCollection(batch_size, df, collection_name):
    collection = client.collections.use(collection_name)
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing comments"):
            text = str(row["translated_text"]) if pd.notna(row["translated_text"]) else str(row["text"])
            sentiment = str(row["absa_sentiments"])
            aspects = str(row["absa_aspects"])
            vector_ = openai_embeddings.embed_query(text)
            object_ = {
                "Comment_ID": i,
                "Text": text,
                "Sentiment": sentiment,
                "Aspects": aspects
            }
            batch.add_object(properties=object_, vector=vector_)

collection_name = "Interstellar_Comments"
batch_size = 100

createCollection(collection_name)
loadDataIntoCollection(batch_size, df, collection_name)

# test
vector_ = openai_embeddings.embed_query("Cooper leaving earth")
collection = client.collections.use(collection_name)
docs = collection.query.near_vector(
    near_vector=vector_,
    limit=3,
    return_metadata=query.MetadataQuery(certainty=True)
)
print([(obj.properties["Text"], obj.metadata.certainty) for obj in docs.objects])

client.close()
print("Done!")