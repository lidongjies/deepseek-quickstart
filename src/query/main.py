import os
import json
from glob import glob
from dotenv import load_dotenv
from pymilvus import MilvusClient
from tqdm import tqdm
from openai import OpenAI
from pymilvus import model as milvus_model

# Load environment variables from .env file
load_dotenv()

def main():
    milvus_client = MilvusClient(
        uri="http://127.0.0.1:19530",
    )
    
    # 获取collection名称
    collection_name = os.getenv("COLLECTION_NAME")

    embedding_model = milvus_model.DefaultEmbeddingFunction()
    
    question = "How is data stored in milvus?"
    
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries([question]),
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
        limit=3,
    )
    
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    
    # print(json.dumps(retrieved_lines_with_distances, indent=4))
    
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
    
    SYSTEM_PROMPT = """
    Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
    """
        
    USER_PROMPT = f"""
    请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    <translated>
    </translated>
    """
    
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
    )
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    
    print(response.choices[0].message.content)

if __name__ == "__main__":
  main()