__version__ = '0.1.0'
__author__ = 'Bruno Evangelista'
__license__ = 'Apache 2.0'

# Setup Logging
# --------------------------------------------------------------------------------
import logging

logging.addLevelName(logging.DEBUG, "\x1b[38;20mDEBUG\x1b[0m")
logging.addLevelName(logging.INFO, "\x1b[32;20mINFO \x1b[0m")
logging.addLevelName(logging.WARNING, "\x1b[33;20mWARN \x1b[0m")
logging.addLevelName(logging.ERROR, "\x1b[31;20mERROR\x1b[0m")


class DefaultLogger(logging.Logger):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def __init__(self, name):
        super().__init__(name)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)-12s %(message)s")
        )
        self.addHandler(self.handler)

    def setLevel(self, level):
        self.handler.setLevel(level)


log = DefaultLogger('kfastml')
log.handler.setLevel(logging.INFO)
