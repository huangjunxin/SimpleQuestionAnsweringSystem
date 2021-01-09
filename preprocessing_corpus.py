import pickle
from utils import low_freq_words_construction, preprocessing_corpus
from analyse_corpus import question_list, words_freq_dict

# 构建低频词库
low_freq_words = low_freq_words_construction(words_freq_dict)

# 对问题列表进行预处理
preprocessed_question_list = preprocessing_corpus(question_list, low_freq_words)
# 保存预处理后的问题列表到文件
with open('model/preprocessed_question_list.pkl', 'wb') as fw:
    pickle.dump(preprocessed_question_list, fw)
