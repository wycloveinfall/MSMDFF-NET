import time


class processBarSelf(object):
    def __init__(self, bar_size=35, decimal=2):
        """
        :param decimal: 保留的保留小数位
        :param bar_size:  #的个数
        """
        self.decimal = decimal
        self.bar_size = bar_size
        self.increase_step = 100 / bar_size  # 在百分比 为几时增加一个 * 号

        self.biase = 50/bar_size*2

        self.start = True
        self.last_perf_counter = time.perf_counter()

    def __call__(self, now, total):
        # 时间管理
        if self.start:
            self.start = False
            self.last_perf_counter = time.perf_counter()

        # 1. 获取当前的百分比数

        percentage = self.percentage_number(now, total)

        # 2. 根据 现在百分比计算
        well_num = int(percentage / self.increase_step)

        # 3. 打印字符进度条
        progress_bar_num = self.progress_bar(well_num)

        # 5. 计算时间

        dur = time.perf_counter() - self.last_perf_counter
        remain = dur/(now+0.001)*(total-now)

        velocity = now/dur

        # 5. 完成的进度条
        result = "\r{:^3.0f}%{} {:d}/{:d} {:.2f}it/s [{:.2f}s {:.2f}s]".format(well_num*self.biase, progress_bar_num,now,total,velocity,dur,remain)
        return result

    def percentage_number(self, now, total):
        """
        计算百分比
        :param now:  现在的数
        :param total:  总数
        :return: 百分
        """
        return round(now / total * 100, self.decimal)

    def progress_bar(self, num):
        """
        显示进度条位置
        :param num:  拼接的  “#” 号的
        :return: 返回的结果当前的进度条
        """
        # 1. "#" 号个数
        well_num = "*" * num

        # 2. 空格的个数
        b = "." * (self.bar_size - num)

        return "[{}->{}]".format(well_num, b)

if __name__ == '__main__':
    index = processBarSelf()

    start = 500
    for i in range(start + 1):
        print(index(i, start), end='')
        time.sleep(0.01)
        # \r 返回本行开头
        # end : python 结尾不加任何操作, 默认是空格
