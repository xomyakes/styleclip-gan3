import logging.config

LOGGING_CONFIGURATION = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "logging.Formatter",
        },
        "generic": {
            "format": "[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "logging.Formatter",
        },
        "generic_formatter": {
            "format": "[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "logging.Formatter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "access_file": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "level": "INFO",
            "filename": "./access.log",
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stderr",
        },
        "error_file": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": "./error.log",
            "level": "WARNING",
        },
    },
    "loggers": {
        "": {
            "handlers": ["error_console", "error_file"], "level": "INFO", "propagate": False
        },
        "root": {
            "handlers": ["error_console", "error_file"], "level": "INFO", "propagate": False
        },
        # "gunicron.error": {
        #     "handlers": ["error_console", "error_file"], "level": "INFO", "propagate": False
        # },
        # "gunicron.access": {
        #     "handlers": ["console", "access_file"], "level": "INFO", "propagate": False
        # },
        "__main__": {
            "handlers": ["error_console", "error_file"], "level": "INFO", "propagate": False
        },
    },
}


def configure_logging():

    logging.config.dictConfig(LOGGING_CONFIGURATION)
