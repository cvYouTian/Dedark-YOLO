# import os
# import logging.config
#
# LOGGING_NAME = 'YOLOv8-lite'
#
#
# def set_logging(name=LOGGING_NAME, verbose=True):
#     """Sets up logging for the given name."""
#     rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
#     level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
#     logging.config.dictConfig({
#         'version': 1,
#         'disable_existing_loggers': False,
#         'formatters': {
#             name: {
#                 'format': '%(message)s'}},
#         'handlers': {
#             name: {
#                 'class': 'logging.StreamHandler',
#                 'formatter': name,
#                 'level': level}},
#         'loggers': {
#             name: {
#                 'level': level,
#                 'handlers': [name],
#                 'propagate': False}}})
#
#
# # Set logger
# set_logging(LOGGING_NAME, verbose=True)  # run before defining LOGGER
# LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)