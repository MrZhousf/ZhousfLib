import sys
import time
import math


class ProcessBar:

    def __init__(self, image_total_count, deal_per_image_time, negative_factor, update_bar_interval_time=1):
        """

        :param image_total_count: 图片总数
        :param deal_per_image_time: 处理单张图片时间
        :param negative_factor: 消极系数
        :param update_bar_interval_time: 更新进度条间隔时间
        """
        self.start_time = time.time()
        self.image_total_count = image_total_count
        self.deal_per_image_time = deal_per_image_time
        self.negative_factor = negative_factor
        self.update_bar_interval_time = update_bar_interval_time

    @staticmethod
    def print_progress_bar(iteration, total, length=50, fill='>'):
        percent = "{0:.1f}".format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f"\r|{bar}| {percent}% Complete")
        sys.stdout.flush()

    def task_is_finished(self, fake_deal_task_time=15):
        """
        查询任务是否结束了
        :param fake_deal_task_time: 设定一个假的提前结束时间，实际业务需查询数据状态
        :return:
        """
        return (time.time() - self.start_time) > fake_deal_task_time

    def finish_process_bar(self, time_total):
        self.print_progress_bar(time_total, time_total)

    def run(self):
        """
                          刷新频率                        n
        progress_bar = —————————————  =  ————————————————————————————————————
                        进度条总长度       单张图片处理时间 * 图片总数量 * 消极系数
        n取值范围：[1, 进度条总长度]
        消极系数：等于1则不起作用，大于1则使进度条慢点走，建议取值 > 1.5
        """
        self.start_time = time.time()
        # 进度条总长度
        bar_total_length = math.ceil(self.deal_per_image_time * self.negative_factor * self.image_total_count)
        # 根据估算时间更新进度条
        for i in range(bar_total_length):
            # 设置进度条更新间隔时间
            time.sleep(self.update_bar_interval_time)
            # 更新进度条
            self.print_progress_bar(i + 1, bar_total_length)
            # 获取任务状态：若任务提前结束了，则提前结束进度条
            if self.task_is_finished():
                self.finish_process_bar(bar_total_length)
                break


if __name__ == "__main__":
    pb = ProcessBar(image_total_count=100,  # 图片总数
                    deal_per_image_time=1,  # 处理单张图片时间
                    negative_factor=1.5,  # 消极系数
                    update_bar_interval_time=1,  # 更新进度条间隔时间
                    )
    pb.run()
    print("")
    print(f"消极系数：{pb.negative_factor}")
    print(f"总耗时：{time.time() - pb.start_time}")
