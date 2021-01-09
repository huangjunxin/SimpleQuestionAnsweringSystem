import pickle
from utils import get_inverted_index, get_full_inverted_index
from analyse_corpus import words_freq_dict

# 导入预处理后的问题列表
preprocessed_question_list = pickle.load(open('model/preprocessed_question_list.pkl', 'rb'))

# 构建倒排索引表
inverted_index = get_inverted_index(words_freq_dict, preprocessed_question_list)
# 保存倒排表结果到文件
with open('model/inverted_index.pkl', 'wb') as fw:
    pickle.dump(inverted_index, fw)

# # 构建全量倒排索引表
# inverted_index = get_full_inverted_index(preprocessed_question_list)
# # 保存倒排表结果到文件
# with open('model/inverted_index_full.pkl', 'wb') as fw:
#     pickle.dump(inverted_index, fw)
