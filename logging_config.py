import logging.config
import json

loggin_config = json.load(open('./logging.json'))
logging.config.dictConfig(loggin_config)

