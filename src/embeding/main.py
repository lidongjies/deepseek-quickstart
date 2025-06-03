import os
from glob import glob
from dotenv import load_dotenv
from pymilvus import MilvusClient
from tqdm import tqdm
from pymilvus import model as milvus_model

# Load environment variables from .env file
load_dotenv()

def main():
    text_lines = []

    for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()
        text_lines += file_text.split("# ")
        
    print(len(text_lines))
    
    # 获取embedding维度
    embedding_model = milvus_model.DefaultEmbeddingFunction()
    test_embedding = embedding_model.encode_queries(["This is a test"])[0]
    embedding_dim = len(test_embedding)
    
    # 初始化milvus客户端
    milvus_client = MilvusClient(
        uri="http://127.0.0.1:19530",
    )
    
    # 获取collection名称
    collection_name = os.getenv("COLLECTION_NAME")
    
    # 如果collection存在，则删除
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # 创建collection
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # 内积距离
        consistency_level="Strong",  # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)。更多详情请参见 https://milvus.io/docs/consistency.md#Consistency-Level。
    )
    
    # 创建embeddings
    data = []
    doc_embeddings = embedding_model.encode_documents(text_lines)
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": doc_embeddings[i], "text": line})
    milvus_client.insert(collection_name=collection_name, data=data)


if __name__ == "__main__":
    main()
