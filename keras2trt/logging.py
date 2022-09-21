import logging

logger = logging.getLogger("keras2trt")
logger.setLevel(logging.INFO)

sh_error = logging.StreamHandler()
sh_error.setLevel(logging.INFO)
logger.addHandler(sh_error)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
for handler in logger.handlers:
    handler.setFormatter(formatter)
