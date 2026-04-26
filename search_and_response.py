"""

向量检索

"""
def vector_search(db_client, emb_model , query):
    """
    查询向量化，并从向量数据库中检索相似向量
    
    参数：
        db_client: 向量数据库
        emb_model: 嵌入模型
        query: 输入向量

    返回：
        context_chunks: 查询到的相关上下文

    """
    # 引入embedding模型
    embedding_model = emb_model

    # 查询向量化
    query_vectors = embedding_model.encode_query([query])

    # 向量检索
    context_chunks = db_client.search(
        collection_name="demo_collection",  # 目标collection
        data=query_vectors,  # 查询向量
        limit=2,  # 返回实体的数量
        output_fields=["page_number", "sentense_chunk"],  # 返回的字段
    )
    print("检索完成！")
    print("查询返回为：", context_chunks)
    return context_chunks

"""

上下文组装&回复生成

"""
def usr_prompt(query, context_chunks):
    """
    使用检索回的上下文信息构建用户提示词

    参数：
        query: 用户提问
        context_chunks: 上下文信息

    返回：
        prompr: 结构化的用户提示词
    
    """

    # 将检索回的上下文信息存入列表
    context_chunks_list = [] 
    for context in context_chunks[0]:
        context_chunks_list.append(context["entity"]["sentense_chunk"])
    # print("上下文格式转换完成！")
    # print("context_chunks_list", context_chunks_list)
    # 构建用户提示词
    # 对检索回来的chunk进行拼接
    fore_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(context_chunks_list)]) 
    prompt = f"{fore_prompt}\nQuestion: {query}"
    return prompt

def generate_response(system_prompt, user_message, client_chat, model="gpt-4o"):
    """
    根据系统提示和用户消息生成AI模型的回复。

    参数:
        system_prompt (str): 指导AI行为的系统提示。
        user_message (str): 用户的消息或查询。
        model (str): 用于生成回复的模型。

    返回:
        dict: AI模型的回复。
    """
    response = client_chat.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
  
    return response

def search_and_response(db_client, emb_client, chat_client, query):
    context_chunks = vector_search(db_client, emb_client, query)

    # 用户提示词组装
    user_prompt = usr_prompt(query, context_chunks)

    # 系统提示词
    system_prompt = "你是一个AI助手，严格根据给定的情境回答问题。如果无法直接从上下文中得出答案，请回复：“我没有足够的信息来回答这个问题。”"

    # 回答生成
    response = generate_response(system_prompt, user_prompt, chat_client, model="gpt-4o")
    response_text = response.choices[0].message.content
    print("context_chunks:", context_chunks)

    # 带原文位置的回答
    refer = ("※回答依据为文档第%d页和第%d页"%(context_chunks[0][0]["page_number"], context_chunks[0][1]["page_number"]))
    response_refer = f"{response_text}\n  \n============================================\n  \n{refer}" # response和context_chunks的数据类型分别是什么

    return response_refer