import logging


class Logger:
    def __init__(self, it_path="./logs/recent/log_iteration_1", sum_path="./logs/recent/log_summary"):
        self.__logger_it = self.__setup_logger("logger_iteration", it_path)
        self.__logger_sum = self.__setup_logger("logger_summary", sum_path)

    def __setup_logger(self, name, file):
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler = logging.FileHandler(file, mode="w+")
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger

    def log(self, msg, to_summary=False, to_iteration=False):
        if to_summary:
            self.__logger_sum.info(msg)
        if to_iteration:
            self.__logger_it.info(msg)

    def set_log_it_file(self, num, file):
        self.__logger_it = self.__setup_logger(f"logger_iteration_{num}", file)
