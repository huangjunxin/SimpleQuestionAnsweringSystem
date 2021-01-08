from sklearn.metrics.pairwise import cosine_similarity
from utils import question_list, answer_list, preprocessing_corpus, word_segmentation
import pickle

# 导入向量化后的问题列表
vectorized_question_list = pickle.load(open('model/vectorized_question_list.pkl', 'rb'))
# 导入向量化器
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
# 导入倒排表索引
inverted_index = pickle.load(open('model/inverted_index.pkl', 'rb'))


def get_answers(input_question):
    # 对输入的问题进行倒排表索引的匹配
    index_list = []
    for sentence in word_segmentation([input_question]):
        for word in sentence:
            if word in inverted_index.keys():
                index_list += inverted_index[word]
    # 对匹配的索引进行去重
    index_list = list(set(index_list))
    # 对输入的问题进行预处理
    preprocessed_input_question = preprocessing_corpus([input_question])
    # 利用向量化器对问题进行向量化
    vectorized_input_question = vectorizer.transform(preprocessed_input_question)
    # 计算余弦相似度
    res = cosine_similarity(vectorized_input_question, vectorized_question_list[index_list])[0]
    res_index = res.argsort()[-5:].tolist()[::-1]
    for index in res_index:
        print(question_list[index], answer_list[index])


get_answers('''In which decade did Beyonce become famous''')
