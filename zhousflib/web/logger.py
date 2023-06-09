# -*- coding: utf-8 -*-
# @Author  : zhousf-a
# @Function: 业务日志记录
import time
from pathlib import Path


class Logger(object):
    def __init__(self, log_dir: Path = None, g=None):
        """
        数据链
        :param log_dir: 日志目录，默认空
        :param g: flask.g 默认空
        :param print: 打印日志
        """
        self.log_dir = log_dir
        day_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())) + '\n'
        if g is not None:
            if hasattr(g, "log"):
                day_time = g.log
            g.logger = self
        # 日志信息-详细
        self.log_txt = day_time
        self.log(log_dir)
        # 日志信息-仅标题
        self.__log_txt_level = []

    def print_log(self):
        print(self.log_txt)

    @property
    def level_log_first(self):
        txt = [title for (level, title) in self.__log_txt_level if level <= 1]
        return "\n".join(txt)

    @property
    def level_log_second(self):
        txt = [title for (level, title) in self.__log_txt_level if level <= 2]
        return "\n".join(txt)

    @property
    def level_log_third(self):
        txt = [title for (level, title) in self.__log_txt_level if level <= 3]
        return "\n".join(txt)

    def title_first(self, title):
        title = "------------ {0} ------------".format(title)
        self.log_txt = '{0}{1}\n'.format(self.log_txt, title)
        self.__log_txt_level.append((1, title))
        return self

    def title_second(self, title):
        title = "****** {0} ******".format(title)
        self.log_txt = '{0}{1}\n'.format(self.log_txt, title)
        self.__log_txt_level.append((2, title))
        return self

    def title_third(self, title):
        title = "【 {0} 】".format(title)
        self.log_txt = '{0}{1}\n'.format(self.log_txt, title)
        self.__log_txt_level.append((3, title))
        return self

    def log(self, msg):
        """
        记录日志
        :param msg: 信息
        :return:
        """
        if msg is not None:
            self.log_txt = '{0}{1}\n'.format(self.log_txt, msg)
        return self

    # noinspection PyBroadException
    def save_log(self):
        """
        保存日志文件
        :return:
        """
        if self.log_dir is None:
            return
        log_file = "{0}/log.txt".format(self.log_dir)
        log_file = log_file.replace("\\", "/")
        day_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time())) + '\n'
        self.log_txt = '{0}\n{1}'.format(self.log_txt, day_time)
        try:
            with open(log_file, "a+", encoding="utf-8") as f:
                f.write(self.log_txt)
        except Exception as e:
            print(e)
            pass
