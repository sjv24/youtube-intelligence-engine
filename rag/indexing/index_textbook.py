import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weaviate
# import ollama  # Option A only
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes import query
# from unstructured.partition.pdf import partition_pdf  # Option A only
from utils.general_functions import getRawPath, saveJson, readJson
from pipeline.embedding import local_embeddings, getLocalEmbedding

client = weaviate.connect_to_local()

# # Option A only - extract textbook
# def extractTextbook():
#     filepath = os.path.join(getRawPath(), "the_science_of_interstellar.pdf")
#     elements = partition_pdf(
#         languages=["eng"],
#         filename=filepath,
#         extract_images_in_pdf=True,
#         strategy="hi_res"
#     )
#     print(f"Extracted {len(elements)} elements")
#     return elements

# # Option A only - caption images using llava
# def captionImage(image_path):
#     try:
#         response = ollama.chat(
#             model="llava",
#             messages=[{
#                 "role": "user",
#                 "content": "Describe this image briefly in context of physics and space science:",
#                 "images": [image_path]
#             }]
#         )
#         return response["message"]["content"]
#     except Exception as e:
#         print(f"Image captioning failed: {e}")
#         return ""

# # Option A only - process elements into chunks
# def processElements(elements):
#     chunks = []
#     for i, element in enumerate(elements):
#         if element.category == "Image":
#             img_path = element.metadata.image_path
#             if img_path:
#                 caption = captionImage(img_path)
#                 if caption:
#                     chunks.append({
#                         "par_num": i,
#                         "title": "The Science of Interstellar",
#                         "chunk": caption,
#                         "type": "image_caption"
#                     })
#         else:
#             text = element.text
#             if text and text.strip():
#                 chunks.append({
#                     "par_num": i,
#                     "title": "The Science of Interstellar",
#                     "chunk": text,
#                     "type": element.category
#                 })
#     return chunks

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
    properties.append(Property(name="Type",
                    data_type=DataType.TEXT,
                    description='Element type - text or image caption',
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

def loadDataIntoCollection(batch_size, data, collection_name):
    collection = client.collections.use(collection_name)
    with collection.batch.fixed_size(batch_size=batch_size) as batch:
        for i, a_chunk in enumerate(data):
            vector_ = getLocalEmbedding(a_chunk["chunk"])
            object_ = {
                "Par_ID": a_chunk["par_num"] + 1,
                "Title": a_chunk["title"],
                "Text": a_chunk["chunk"],
                "Type": a_chunk.get("type", "NarrativeText")  # Option B default
            }
            batch.add_object(properties=object_, vector=vector_)
            if i % 100 == 0:
                print(f"Indexed {i}/{len(data)} chunks...")

collection_name = "Interstellar_Textbook"
batch_size = 10

data = readJson("textbook_sents.json")

print("Creating Weaviate collection...")
createCollection(collection_name)
loadDataIntoCollection(batch_size, data, collection_name)

# test
vector_ = getLocalEmbedding("black hole gravity")
collection = client.collections.use(collection_name)
docs = collection.query.near_vector(
    near_vector=vector_,
    limit=3,
    return_metadata=query.MetadataQuery(certainty=True)
)
print([(obj.properties["text"], obj.metadata.certainty) for obj in docs.objects])

client.close()
print("Done!")