import logging


class Logger:
    def __init__(self, it_path="./logs/recent/log_it_1", sum_path="./logs/recent/log_summary"):
        self.logger_it = self.__setup_logger("logger_it", it_path)
        self.logger_sum = self.__setup_logger("logger_sum", sum_path)

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
            self.logger_sum.info(msg)
        if to_iteration:
            self.logger_it.info(msg)

    def set_log_it_file(self, num, file):
        self.logger_it = self.__setup_logger(f"logger_it_{num}", file)
