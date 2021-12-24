from functools import wraps
import time
import inspect
import logging


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__} took {elapsed_time} seconds.")
        return result
    return wrapper


class CustomFilter(logging.Filter):
    def filter(self, record):
        record.real_filename = getattr(record,
                                       'real_filename',
                                       record.filename)
        record.real_funcName = getattr(record,
                                       'real_funcName',
                                       record.funcName)
        record.real_lineno = getattr(record,
                                     'real_lineno',
                                     record.lineno)
        return True


def get_logger():
    log_format = '[%(asctime)s] %(levelname)s\t%(real_filename)s' \
                 ' - %(real_funcName)s:%(real_lineno)s -> %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addFilter(CustomFilter())
    return logger


def log(logger):
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            extra = {'real_filename': inspect.getfile(func),
                     'real_funcName': func_name,
                     'real_lineno': inspect.currentframe().f_back.f_lineno}
            logger.info(f'[START] {func_name}', extra=extra)
            try:
                return func(*args, **kwargs)
            except Exception as err:
                logging.error(err, exc_info=True, extra=extra)
                logging.error(f'[KILLED] {func_name}', extra=extra)
            else:
                logging.info(f'[END] {func_name}', extra=extra)
        return wrapper
    return _decorator
