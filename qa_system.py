from qa_function import get_answers, get_answers_glove, get_answers_optimized, get_answers_glove_optimized, inverted_index
from utils import question_list

# print(inverted_index)

# res_list = get_answers('''Whose grave isn't the only one in the abbey on which it is encouraged to walk?''')
# for index in res_list:
#     print(question_list[index])
# print('----------')

res_list = get_answers_optimized('''In which decade did Beyonce become famous''')
for index in res_list:
    print(question_list[index])
print('----------')
# get_answers_optimized('''What areas did Beyonce compete in when she was growing up?''')
# print('----------')
# get_answers_optimized('''What was the latest version of iTunes as of mid-2015?''')
# print('----------')
# get_answers_optimized('''What products were exported along with indigo from the Lowcountry?''')
# print('----------')
# get_answers_optimized('''What supply port was opened late in 1944?''')

# res_list = get_answers_glove('''Whose grave isn't the only one in the abbey on which it is encouraged to walk?''')
# for index in res_list:
#     print(question_list[index])
# print('----------')
#
# res_list = get_answers_glove_optimized('''Whose grave isn't the only one in the abbey on which it is encouraged to walk?''')
# for index in res_list:
#     print(question_list[index])
# print('----------')
# get_answers_glove_optimized('''What areas did Beyonce compete in when she was growing up?''')
# print('----------')
# get_answers_glove_optimized('''What was the latest version of iTunes as of mid-2015?''')
# print('----------')
# get_answers_glove_optimized('''What products were exported along with indigo from the Lowcountry?''')
# print('----------')
# get_answers_glove_optimized('''What supply port was opened late in 1944?''')
