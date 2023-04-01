import logging


class Logger:
    def __init__(self, it_path="../logs/recent/log_it_1", sum_path="../logs/recent/log_summary"):
        self.logger_it = self.setup_logger("logger_it", it_path)
        self.logger_sum = self.setup_logger("logger_sum", sum_path)

    def setup_logger(self, name, file):
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler = logging.FileHandler(file, mode="w+")
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger

    def log_sum(self, msg):
        self.logger_sum.info(msg)

    def log_it(self, msg):
        self.logger_it.info(msg)

    def log_both(self, msg):
        self.logger_it.info(msg)
        self.logger_sum.info(msg)

    def set_log_it_file(self, num, file):
        self.logger_it = self.setup_logger(f"logger_it_{num}", file)
