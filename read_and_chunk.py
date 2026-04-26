"""

文件地址

"""
pdf_path = "Satelite_remote_sensing_Al.pdf"

""""

文件处理

"""
import fitz 
from tqdm.auto import tqdm 


def text_formatter(text: str) -> str:# 去除回车和开头结尾后的空格
    """对文本进行轻微的格式化处理"""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    打开PDF文件，逐页阅读其文本内容，并收集统计数据。

    参数:
        pdf_path (str): PDF文档的文件路径，要打开并读取。

    返回:
        list[dict]: 一个词典列表，每条都包含页码、字数、句数、token数以及每页提取的文本。
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # 迭代文档页面
        text = page.get_text() # 调用PyMuPDF中Page对象的get_text()方法，获取纯文本编码为UTF-8
        text = text_formatter(text) #调用先前定义的轻微格式化函数 去除回车和开头结尾后的空格
        pages_and_texts.append({"page_number": page_number + 1,  # 调整文档页码
                                "page_char_count": len(text), # 每页字数
                                "page_sentence_count_raw": len(text.split("。")),
                                "page_token_count": len(text) * 1.5, # 按1.5倍汉字数估算token
                                "text": text})
    return pages_and_texts


# print("pages_and_chunks[40]", pages_and_chunks[40])

"""

文本切分

"""
def pages_to_sentences(pages, sentence_model):
    """
    将每一页的句子进行切分，并保存至句子列表，作为新增项添加到原列表中

    参数：
        pages: 按页划分的文档文本及信息
        sentence_model: 实现句子切分使用的nlp模型

    返回：
        list[dict]: 每个列表元素包含页码、字数、句数、token数、每页提取的文本、句子列表项、句子数量
    """

    # 将每一页的句子进行切分
    for item in tqdm(pages):
        # 将text切分成句子列表 并存储到sentences
        item["sentences"] = list(sentence_model(item["text"]).sents)
        
        # 保证所有句子都是字符串
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        
        # 保存每一页的句子数量 
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages

def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]: #slice_size: 切块大小
        """将句子列表切块"""
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)] # 将大列表切分成多个固定大小的小列表 
def pages_sentences_to_chunks(pages_sentences, chunk_size):
    """
    将每页的文本按句子数切分成文本块，并保存至句子块列表，作为新增项添加到原列表中

    参数：
        pages_sentences: 含有句子列表和句子数量项的按页划分的文档文本及信息
        chunk_size: 每个文本块最多含有几个句子

    返回：
        list[dict]: 每个列表元素包含页码、字数、句数、token数、每页提取的文本、句子列表项、句子数量、句子块列表、句子块数量
    
    """
    
    for item in tqdm(pages_sentences): # 对pages_and_texts中的sentences
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            slice_size=chunk_size)# 切片对象：句子列表， 切片大小:6, 切片结果: n个长度为6的列表
        item["num_chunks"] = len(item["sentence_chunks"]) # 切块数量/列表sentence_chunks的长度
    return pages_sentences

def pages_to_chunks(pages_sentences_chunks):
    """
    将文档文本按句子块进行划分，存入新的列表中

    参数：
        pages_sentences_chunks(list[dict]): 含有句子块列表和句子块数量项的按页划分的文档文本及信息

    返回：
        chunks(list[dict]): 每个列表元素含有句子块所在页码、句子块文本、句子块字数、句子块的token数

    """
    chunks = []
    for item in tqdm(pages_sentences_chunks):
        for sentence_chunk in item["sentence_chunks"]: # 对sentence_chuncks中的每个句子块进行操作
            chunk_dict = {} # 创建字典
            chunk_dict["page_number"] = item["page_number"] # page_number项存放页码
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip() # 将句子块中的句子连成段落
            chunk_dict["sentence_chunk"] = joined_sentence_chunk # 将拼接好的句子段落存入sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk) # sentence_chunk的字符数量
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) * 1.5 # 1 token = ~4 characters # sentence_chunk的token数
            chunks.append(chunk_dict) # 将每个句子块的字典存入新建列表pages_and_chuncks
    return chunks

def chunking(pdf_path, sentence_model, chunk_size):
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_sentences = pages_to_sentences(pages_and_texts, sentence_model)
    pages_sentences_chunks = pages_sentences_to_chunks(pages_sentences, chunk_size)
    chunks = pages_to_chunks(pages_sentences_chunks)
    return chunks




if __name__ == '__main__':
    import pandas as pd
    import spacy

    # 测试open_and_read_pdf
    pages = open_and_read_pdf(pdf_path=pdf_path)
    print(pages[12]) # 显示前两页信息
    # nlp = spacy.blank("zh")
    # nlp.add_pipe("sentencizer")
    # # 测试pages_to_sentences
    # pages_sentences = pages_to_sentences(pages, nlp)
    # # print("pages_sentences[10]", pages_sentences[10])
    # # 测试pages_sentences_to_chunks
    # pages_sentences_chunks = pages_sentences_to_chunks(pages_sentences, 6)
    # # print(len(pages_sentences_chunks))
    # # 测试pages_to_chunks
    # chunks = pages_to_chunks(pages_sentences_chunks)
    # print("len(chunks)", len(chunks))
    # print("chunks[40]", chunks[40])


    # # 以DataFrame格式打印文本信息
   
    # df = pd.DataFrame(pages_and_texts)
    # print(df.head)
    # # 打印统计信息
    # print(df.describe().round(2))
    


