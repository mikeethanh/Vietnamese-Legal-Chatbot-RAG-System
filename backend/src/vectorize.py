import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)
client = QdrantClient(url="http://qdrant-db:6333")


def create_collection(name):
    return client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )


def add_vector(collection_name, vectors={}):
    points = [PointStruct(id=k, vector=v['vector'], payload=v['payload']) for k, v in vectors.items()]
    return client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )


def search_vector(collection_name, vector, limit=4):
    res = client.search(
        collection_name=collection_name, query_vector=vector, limit=limit
    )
    payloads = [x.payload for x in res]
    return payloads
