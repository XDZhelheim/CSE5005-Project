from urllib.parse import urlparse
import os
from ping3 import ping
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import norm
from scipy.stats import kstest


# ping命令获得网络延时
def pingDelay(url):
    # url = urlparse(url).netloc
    cmd = 'ping -n 1 ' + url
    p = os.popen(cmd)
    x = p.read()
    print(x)
    # if '请求超时' in x:
    #     delay = -1
    # # 错误处理
    # if '请求找不到主机' in x:
    #     delay = -1
    # indexerror处理
    try:
        delay = x.split('平均 = ')[1].split('ms')[0]
    except IndexError:
        delay = -1
        return delay
    delay = x.split('平均 = ')[1].split('ms')[0]
    return delay

# 获取resultLink里所有url的延时
def getDelay(resultLink):
    success = 0
    fail = 0
    delayList = []
    for url in resultLink:
        for i in range(10):
            delay = pingDelay(url)
            if delay == -1:
                fail += 1
                break
            else:
                success += 1
            delayList.append(delay)
        print('success: ', success)
        print('fail: ', fail)

    return delayList

# def getDelay2(resultLink):
#     delayList = []
#     for url in resultLink:
#         delay = ping(url)
#         delayList.append(delay)
#     return delayList


# 输出延时列表到txt
def outputDelay(delayList):
    with open('delaymore.txt', 'w') as f:
        for delay in delayList:
            f.write(str(delay) + '\n')

# 将resultLink里所有url读入列表
def readResultLink():
    # resultLink = []
    # with open('resultLink.txt', 'r') as f:
    #     for line in f:
    #         resultLink.append(line.strip())
    with open("../data/links_filtered.json", "r") as f:
        resultLink=json.load(f)
    return resultLink


# step1: 测试延时数据，输出到delay.txt
# resultLink = readResultLink()
# delayList = getDelay(resultLink)
# outputDelay(delayList)

# 去掉延时列表中的-1，并输出到delayDrop.txt
def dropDelay(delayList):
    delayListDrop = []
    for delay in delayList:
        if delay != str(-1):
            delayListDrop.append(delay)
    with open("delaymoreDrop.txt", 'w') as f:
        for delay in delayListDrop:
            f.write(str(delay) + '\n')
    return delayListDrop


def readDelay(delayFile):
    delayList = []
    with open(delayFile, 'r') as f:
        for line in f:
            delayList.append(line.strip())
    return delayList

# step2: 去掉延时列表中的-1，并输出到delayDrop.txt
# delayList = readDelay('delay.txt')
# delayListDrop = dropDelay(delayList)

delayList = readDelay('delaymore.txt')
delayListDrop = dropDelay(delayList)
# 将列表中的延时转换为int类型
delayListDrop = [int(delay) for delay in delayListDrop]
# 求列表数据均值，标准差
mean = np.mean(delayListDrop)
std = np.std(delayListDrop)
print('mean: ', mean)
print('std: ', std)

# 给出均值，标准差，生成正态分布，并画图
def normalDistribution(mean, std):
    # 生成正态分布
    x = np.linspace(mean - 3 * std, mean + 3 * std, 50)
    y = norm.pdf(x, mean, std)
    # 画图
    plt.plot(x, y)
    plt.show()

# normalDistribution(mean, std)

# 对正态分布作ks检验
def ksTest(delayListDrop, mean, std):
    # 生成正态分布
    x = np.linspace(mean - 3 * std, mean + 3 * std, 50)
    y = norm.pdf(x, mean, std)
    print(kstest(delayListDrop, 'norm', args=(mean, std)))

# ksTest(delayListDrop, mean, std)

# 利用正态分布生成随机数
def randomDelay(mean, std):
    delay = np.random.normal(mean, std)
    # 生成的随机数不能小于0,如果小于0，重新生成
    while delay < 0:
        delay = randomDelay(mean, std)
    return delay

# 基于正态分布生成n个随机延迟
def randomDelayList(mean, std, n):
    delayOUTList = []
    for i in range(100):
        delay = randomDelay(mean, std)
        delayOUTList.append(delay)
    return delayOUTList

print(randomDelayList(mean, std, 100)) # 单位ms
#mean = 114.113
# std = 81.4416




