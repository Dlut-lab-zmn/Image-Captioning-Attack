import logging
from logging import handlers
import sys
import time

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)
if __name__ == '__main__':
    #log = Logger('all.log',level='debug')
    #log.logger.debug('debug')
    #log.logger.info('info')
    #log.logger.warning('warning')
    #log.logger.error('error')
    #log.logger.critical('serere')
    #Logger('error.log', level='error').logger.error('error')
    f=open("myprint.txt","w+")
    sys.stdout=f
    for i in range(1000):
        print(i*9)
    
    
    print("dsda")