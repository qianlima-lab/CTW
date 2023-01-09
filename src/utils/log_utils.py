import logging
import os
import sys
import re
from logging.handlers import RotatingFileHandler

def create_logfile(args,filename):
    project_path = re.compile(r'.*CTW_code').findall(os.getcwd())[0]
    logdir_ = args.outfile.split('.')[0]
    logdir = os.path.join(project_path,'logfiles',logdir_)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir,filename)
    return logfilename

def get_logger(level=logging.INFO, debug=False,args=None,filename='logfile.log'):
    # create logger
    log = logging.getLogger()
    log.setLevel(level=level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    if not debug:
        # create file handler for logger.
        fh = RotatingFileHandler(create_logfile(args,filename),
                                 maxBytes=10485760, backupCount=10,
                                 encoding='utf-8')
        fh.setLevel(level=level)
        fh.setFormatter(formatter)
    # create console handler for logger.
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if not debug:
        log.addHandler(fh)

    # log.addHandler(ch)
    return log


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        # keep the console
        if self.log_level != logging.INFO:
            self.terminal.write('\033[31m' + buf + '\033[0m')
        else:
            self.terminal.write(buf)

        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''
