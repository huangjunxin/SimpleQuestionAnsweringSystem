import pickle
import numpy as np
from utils import get_sentence_vector, get_glove_matrix

# 导入预处理后的问题列表
preprocessed_question_list = pickle.load(open('model/preprocessed_question_list.pkl', 'rb'))

# # 建立GloVe矩阵
# glove_words, embeddings = get_glove_matrix()
#
# # 保存GloVe中的词头到文件
# with open('model/glove_words.pkl', 'wb') as fw:
#     pickle.dump(glove_words, fw)
#
# # 保存矩阵化后的GloVe到文件
# with open('model/glove_embeddings.pkl', 'wb') as fw:
#     pickle.dump(embeddings, fw)

# 导入GloVe中的词头
glove_words = pickle.load(open('model/glove_words.pkl', 'rb'))

# 导入GloVe矩阵
embeddings = pickle.load(open('model/glove_embeddings.pkl', 'rb'))

# 初始化向量化后的问题列表
vectorized_question_list = []
for preprocessed_sentence in preprocessed_question_list:
    vectorized_sentence = get_sentence_vector(preprocessed_sentence, glove_words, embeddings)
    vectorized_question_list.append(vectorized_sentence)
vectorized_question_list = np.asarray(vectorized_question_list)

# 保存向量化后的问题列表结果到文件
with open('model/vectorized_question_list(glove).pkl', 'wb') as fw:
    pickle.dump(vectorized_question_list, fw)
