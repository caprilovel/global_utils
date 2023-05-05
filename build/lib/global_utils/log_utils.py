import sys,os,time
from datetime import datetime 
import pytz

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

class Logger(object):
    def __init__(self, logFile='./Default.log'):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')
        self.close_flag = False
    
    def write(self, message):
        self.terminal.write(message)
        if self.close_flag is False:
            self.log.write(message)
    
    def flush(self):
        pass
    
    def close(self):
        self.log.close()
        self.close_flag = True

def get_time_str(style='Nonetype'):
    t = time.localtime()
    if style is 'Nonetype':
        return ("{}{}{}{}{}{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    elif style is 'underline':
        return ("{}_{}_{}_{}_{}_{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def timestamp():
    time_format = '%Y%m%d%H%M'
    timer = datetime.now(pytz.timezone('Asia/Shanghai')).strftime(time_format)
    return timer