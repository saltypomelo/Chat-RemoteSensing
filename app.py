import gradio as gr
from search_and_response import search_and_response
from embed_and_store import embed_and_store
from read_and_chunk import pdf_path, chunking
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import spacy
import os
import yaml

# 读取Milvus配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 取出配置值
milvus_host = config["milvus"]["host"]
milvus_port = config["milvus"]["port"]


db_client  = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
emb_client = SentenceTransformer('BAAI/bge-m3')
chat_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
nlp = spacy.blank("zh")
nlp.add_pipe("sentencizer")

# 将文档进行chunking
chunks = chunking(pdf_path, nlp, 6)
# 将文本块向量化并存入向量数据库
embed_and_store(emb_client, db_client,chunks)

def main(query):
    response_refer = search_and_response(db_client, emb_client, chat_client, query)
    return response_refer
    
demo = gr.Interface(
    fn=main,
    inputs=["text"],
    outputs=["text"],
    api_name="predict"
)

demo.launch()