from qa_function import get_answers, get_answers_glove, get_answers_optimized, get_answers_glove_optimized
from utils import question_list
import time
import random


# 单次性能测试函数
def single_performance_test(question_index, test_get_answers):
    # 初始化是否命中
    is_correct = 0
    # 性能测试
    start = time.perf_counter()
    res_index_get_answers = test_get_answers(question_list[question_index])
    end = time.perf_counter()
    # 耗时统计
    time_consuming = end - start

    if question_index in res_index_get_answers:
        is_correct = 1
    # else:
    #     print(question_list[question_index])
    return is_correct, time_consuming


# 设置测试次数
test_times = 5
# 初始化函数名
func_name = ['get_answers', 'get_answers_glove', 'get_answers_optimized', 'get_answers_glove_optimized']
# 初始化命中计数，0：get_answers；1：get_answers_glove；2：get_answers_optimized；3：get_answers_glove_optimized
is_correct = [0, 0, 0, 0]
# 初始化时间统计，0：get_answers；1：get_answers_glove；2：get_answers_optimized；3：get_answers_glove_optimized
time_consuming = [0, 0, 0, 0]
for i in range(test_times):
    # print(len(question_list)) 130319
    # 生成随机数
    random_question_index = random.randint(0, 130318)
    # 测试离散式方法问答函数
    res_is_correct, res_time_consuming = single_performance_test(random_question_index, get_answers)
    is_correct[0] += res_is_correct
    time_consuming[0] += res_time_consuming
    # 测试分布式方法问答函数
    res_is_correct, res_time_consuming = single_performance_test(random_question_index, get_answers_glove)
    is_correct[1] += res_is_correct
    time_consuming[1] += res_time_consuming
    # 测试离散式方法优化问答函数
    res_is_correct, res_time_consuming = single_performance_test(random_question_index, get_answers_optimized)
    is_correct[2] += res_is_correct
    time_consuming[2] += res_time_consuming
    # 测试分布式方法优化问答函数
    res_is_correct, res_time_consuming = single_performance_test(random_question_index, get_answers_glove_optimized)
    is_correct[3] += res_is_correct
    time_consuming[3] += res_time_consuming

print(is_correct)
print(time_consuming)
for func in range(4):
    print('Function name: ' + func_name[func])
    print('Correct times: ' + str(is_correct[func]) + ' / ' + str(test_times))
    print('Hit rate: ' + str(is_correct[func] / test_times))
    print('Total time consuming: ' + str(time_consuming[func]))
    print('Average time consuming: ' + str(time_consuming[func] / test_times))
    print('----------')
