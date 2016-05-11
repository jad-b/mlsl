import logging
import logging.config


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'simple': {
            'format': '%(asctime)s [%(levelname)s] %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        }
    },
    'loggers': {
        'mlsl': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'mlsl.performance': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'mlsl.testing': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
})

# 'log' is convenient alias. 79 char lines can be tough to hit. Plus, it
# reminds me of Go.
log = logger = logging.getLogger('mlsl')
perflog = logging.getLogger('mlsl.performance')
testlog = logging.getLogger('mlsl.testing')
