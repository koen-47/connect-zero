import logging


class Logger:
    def __init__(self, iteration_path=None, summary_path=None, experiment_path=None):
        if iteration_path is not None:
            self.__logger_iteration = self.__setup_logger("logger_iteration", iteration_path)
        if summary_path is not None:
            self.__logger_summary = self.__setup_logger("logger_summary", summary_path)
        if experiment_path is not None:
            self.__logger_experiment = self.__setup_logger("logger_experiment", experiment_path)

    def __setup_logger(self, name, file):
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler = logging.FileHandler(file, mode="w+")
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger

    def log(self, msg, to_summary=False, to_iteration=False, to_experiment=False):
        if to_summary:
            self.__logger_summary.info(msg)
        if to_iteration:
            self.__logger_iteration.info(msg)
        if to_experiment:
            self.__logger_experiment.info(msg)

    def set_log_iteration_file(self, num, file):
        self.__logger_iteration = self.__setup_logger(f"logger_iteration_{num}", file)

    def set_log_experiment_file(self, name, file):
        self.__logger_experiment = self.__setup_logger(name, file)
