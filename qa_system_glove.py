from utils import question_list, answer_list, preprocessing_corpus, word_segmentation, get_sentence_vector
import pickle
import numpy as np

# 导入GloVe向量化后的问题列表
vectorized_question_list = pickle.load(open('model/vectorized_question_list(glove).pkl', 'rb'))
# 导入向量化器
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
# 导入倒排表索引
inverted_index = pickle.load(open('model/inverted_index.pkl', 'rb'))
# 导入GloVe中的词头
glove_words = pickle.load(open('model/glove_embeddings.pkl', 'rb'))
# 导入GloVe矩阵
embeddings = pickle.load(open('model/glove_embeddings.pkl', 'rb'))


def get_answers(input_question):
    abs_vec = abs(vectorized_question_list[0])
    # 初始化结果索引列表
    res = []
    # 对输入的问题进行倒排表索引的匹配
    index_list = []
    for sentence in word_segmentation([input_question]):
        for word in sentence:
            if word in inverted_index.keys():
                print('Ok: ' + str(word))
                index_list += inverted_index[word]
    # 对匹配的索引进行去重
    index_list = list(set(index_list))
    # 对输入的问题进行预处理
    preprocessed_input_question = preprocessing_corpus([input_question])
    # 计算输入的问题的向量
    vectorized_input_question = get_sentence_vector(preprocessed_input_question[0], glove_words, embeddings)
    # 遍历倒排表内的所有相关索引值，将相关的向量化后的问题与输入的问题的向量对比，绝对值差小的数存入结果索引列表
    for vectorized_question in vectorized_question_list[index_list]:
        if abs(vectorized_input_question - vectorized_question) < abs_vec:
            abs_vec = abs(vectorized_input_question - vectorized_question)
            res.append(vectorized_question_list[index_list].tolist().index(vectorized_question))
    res_index = np.argsort(res)[-5:].tolist()[::-1]
    for index in res_index:
        print(question_list[index], answer_list[index])


get_answers('''Who predicted that Beyoncé would become the highest paid black entertainer?''')
