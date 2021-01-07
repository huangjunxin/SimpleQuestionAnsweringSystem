import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 导入预处理后的问题列表
preprocessed_question_list = pickle.load(open('model/preprocessed_question_list.pkl', 'rb'))

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer()

# 对问题列表进行向量化
vectorized_question_list = vectorizer.fit_transform(preprocessed_question_list)

# 保存向量化后的问题列表结果到文件
with open('model/vectorized_question_list.pkl', 'wb') as fw:
    pickle.dump(vectorized_question_list, fw)

# 保存向量化器
with open('model/vectorizer.pkl', 'wb') as fw:
    pickle.dump(vectorizer, fw)
