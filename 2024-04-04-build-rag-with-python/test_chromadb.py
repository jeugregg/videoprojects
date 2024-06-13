"""
Just test chromaDB in local disk
"""
import chromadb

relative_path_db = "../vectordb-stores/chromadb"
collectionname = "test01"


client = chromadb.PersistentClient(path=relative_path_db)

if any(collection.name == collectionname for collection in client.list_collections()):
    client.delete_collection(collectionname)
    print('deleting collection')

collection = client.create_collection(name=collectionname)

print(client.list_collections())

assert any(collection.name == collectionname for collection in client.list_collections()), "Collection not created !"
print("TEST OK")