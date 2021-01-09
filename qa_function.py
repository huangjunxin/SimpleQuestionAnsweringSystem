from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocessing_corpus, get_sentence_vector
import pickle
import numpy as np

## 离散式方法
# 导入向量化后的问题列表
vectorized_question_list = pickle.load(open('model/vectorized_question_list.pkl', 'rb'))
# 导入向量化器
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
## 分布式方法
# 导入GloVe向量化后的问题列表
vectorized_question_list_glove = pickle.load(open('model/vectorized_question_list_glove.pkl', 'rb'))
# 导入GloVe中的词头
glove_words = pickle.load(open('model/glove_words.pkl', 'rb'))
# 导入GloVe矩阵
embeddings = pickle.load(open('model/glove_embeddings.pkl', 'rb'))
## 倒排表优化
# 导入倒排表索引
inverted_index = pickle.load(open('model/inverted_index.pkl', 'rb'))


# 离散式方法问答函数
def get_answers(input_question):
    # 对输入的问题进行预处理
    preprocessed_input_question = preprocessing_corpus([input_question])
    # 利用向量化器对问题进行向量化
    vectorized_input_question = vectorizer.transform(preprocessed_input_question)
    # 计算余弦相似度
    res = cosine_similarity(vectorized_input_question, vectorized_question_list)[0]
    res_index = res.argsort()[-5:].tolist()[::-1]
    # for index in res_index:
    #     print(question_list[index], answer_list[index])
    return res_index


# 分布式方法问答函数
def get_answers_glove(input_question):
    # 对输入的问题进行预处理
    preprocessed_input_question = preprocessing_corpus([input_question])
    # 计算输入的问题的向量
    vectorized_input_question = get_sentence_vector(preprocessed_input_question[0], glove_words, embeddings, 100)
    # 计算余弦相似度
    res = cosine_similarity([np.asarray(vectorized_input_question)], vectorized_question_list_glove)[0]
    res_index = res.argsort()[-5:].tolist()[::-1]
    # for index in res_index:
    #     print(question_list[index], answer_list[index])
    return res_index


# 离散式方法优化问答函数
def get_answers_optimized(input_question):
    # 对输入的问题进行预处理
    preprocessed_input_question = preprocessing_corpus([input_question])
    # 对预处理后的问题进行倒排表索引的匹配
    index_list = []
    for sentence in preprocessed_input_question:
        for word in sentence.split(' '):
            if word in inverted_index.keys():
                index_list += inverted_index[word]
    # 对匹配的索引进行去重
    index_list = list(set(index_list))
    # 利用向量化器对问题进行向量化
    vectorized_input_question = vectorizer.transform(preprocessed_input_question)
    # 计算余弦相似度
    if len(index_list) == 0:
        res = cosine_similarity(vectorized_input_question, vectorized_question_list)[0]
    else:
        res = cosine_similarity(vectorized_input_question, vectorized_question_list[index_list])[0]
    res_index = res.argsort()[-5:].tolist()[::-1]
    # for index in res_index:
    #     print(question_list[index], answer_list[index])
    return res_index


# 分布式方法优化问答函数
def get_answers_glove_optimized(input_question):
    # 对输入的问题进行预处理
    preprocessed_input_question = preprocessing_corpus([input_question])
    # 对预处理后的问题进行倒排表索引的匹配
    index_list = []
    for sentence in preprocessed_input_question:
        for word in sentence.split(' '):
            if word in inverted_index.keys():
                index_list += inverted_index[word]
    # 对匹配的索引进行去重
    index_list = list(set(index_list))
    # 计算输入的问题的向量
    vectorized_input_question = get_sentence_vector(preprocessed_input_question[0], glove_words, embeddings, 100)
    # 计算余弦相似度
    if len(index_list) == 0:
        res = cosine_similarity([np.asarray(vectorized_input_question)], vectorized_question_list_glove)[0]
    else:
        res = cosine_similarity([np.asarray(vectorized_input_question)], vectorized_question_list_glove[index_list])[0]
    res_index = res.argsort()[-5:].tolist()[::-1]
    # for index in res_index:
    #     print(question_list[index], answer_list[index])
    return res_index
