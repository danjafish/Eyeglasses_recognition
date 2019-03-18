import time
import sys
from root_service.get_answer import GetAnswerByPath


if __name__ == '__main__':
    answer_getter = GetAnswerByPath()
    start_time = time.time()
    path = sys.argv[1]
    answer_getter.get_answer(path)
    print("--- %s seconds ---" % (time.time() - start_time))
