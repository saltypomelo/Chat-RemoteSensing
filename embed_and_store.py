from tqdm import tqdm
"""

向量化

"""
def embedding(emb_model, chunks):
    """
    
    将切分好的文本块进行向量化处理
    
    参数：
        emb_model: 用于向量化的嵌入模型
        chunks(list[dict]): 切分好的文本块，字典列表，字典项包括页码、句子块文本、句子块字数、句子块token数

    返回：
        list[dict]: 存放向量化结果的字典列表，字典项包含页码、句子块文本、句子块字数、句子块token数、句子块对应的语义向量

    """
    emb_model.to("cpu")

    for item in tqdm(chunks):
        item["embedding"] = emb_model.encode(item["sentence_chunk"])
    return chunks

"""

存入向量数据库

"""

def chunks_to_entities(db_client, chunks_embeddings):
    """
    
    将含有语义向量项的数据转换成符合Milvus要求的实体格式

    参数：
        db_client: 向量数据库客户端
        chunks_embeddingslist[dict]: 存放向量化结果的字典列表，字典项包含页码、句子块文本、句子块字数、句子块token数、句子块对应的语义向量

    返回：
        list[dict]: 字典列表，字典项按顺序为：id、向量、所在页码、句子块文本
    """
    # 创建Collections
    if db_client.has_collection(collection_name="demo_collection"):
        db_client.drop_collection(collection_name="demo_collection")
    db_client.create_collection(
        collection_name="demo_collection",
        dimension=1024,  
    )

    # 将pages_and_chuncks修改为['id', 'vectores', 'page_number', 'sentence_chunk']格式
    entities= []
    for i, item in enumerate(chunks_embeddings):
        entity = {}
        entity["id"] = i
        entity["vector"] = item["embedding"]
        entity["page_number"] = item["page_number"]
        entity["sentense_chunk"] = item["sentence_chunk"]
        entities.append(entity)
    return entities

def embed_and_store(emb_client, db_client, chunks):
    chunks_embedding = embedding(emb_client, chunks) 
    entities = chunks_to_entities(db_client,  chunks_embedding)
    db_client.insert(collection_name="demo_collection", data=entities)
    return entities


if __name__ == '__main__':
    #  ------------------向量化------------------------
    from sentence_transformers import SentenceTransformer
    from read_and_chunk import open_and_read_pdf, pages_to_sentences, pages_sentences_to_chunks, pages_to_chunks
    import pandas as pd
    import spacy
    from pymilvus import MilvusClient


    pdf_path = "Satelite_remote_sensing_Al.pdf"
    # 测试open_and_read_pdf
    pages = open_and_read_pdf(pdf_path=pdf_path)
    nlp = spacy.blank("zh")
    nlp.add_pipe("sentencizer")
    # 测试pages_to_sentences
    pages_sentences = pages_to_sentences(pages, nlp)
    # 测试pages_sentences_to_chunks
    pages_sentences_chunks = pages_sentences_to_chunks(pages_sentences, 6)
    # 测试pages_to_chunks
    chunks = pages_to_chunks(pages_sentences_chunks)
    # 测试embedding
    embedding_model = SentenceTransformer('BAAI/bge-m3')
    chunks_embedding = embedding(embedding_model, chunks)
    # print("chunks_embedding[40]", chunks_embedding[40])
    # 测试chunks_to_entities
    milvus_client = MilvusClient(uri="http://localhost:19530")
    entities = chunks_to_entities(milvus_client,  chunks_embedding)
    print("entities[40]: ", entities[40])
   
   
