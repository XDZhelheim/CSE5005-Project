import requests
from bs4 import BeautifulSoup

url = ["https://youquhome.com/", "https://www.baidu.com/", "https://www.jinshow.net/", "https://www.163.com/", "https://www.sina.com.cn/", "https://www.fenleimulu.net/siteinfo/14613.html", "https://www.wangzhanchi.com/", "https://www.wangzhanchi.com/","https://zhuanlan.zhihu.com/p/51787793", "https://www.laikanxia.com/asia/china/view/12370.html", "https://www.0dh.cn/", "https://blog.csdn.net/z136370204/article/details/104992466/", "https://www.99876.cn/", "https://www.023dir.com/website/29259.html", "https://zhuanlan.zhihu.com/p/390581323", "https://zhuanlan.zhihu.com/p/401554060", "https://www.zhihu.com/question/20354376/answer/14914375", "https://sz.58.com/", "https://www.hao268.com/", "https://www.mihugou.com/", "http://www.1234l.com/", "https://www.zhihu.com/question/542729533/answer/2720567581", "https://www.zhihu.com/question/398193048/answer/1410608969", "https://www.sohu.com/", "http://wap.hao123.com/", "https://123.sogou.com/", "http://www.zzdh.net/wzfl/fl-3-445.html", "https://www.2345.com/", "https://www.ezkt.cn/", "http://www.urlglobalsubmit.com/", "http://www.9zwz.com/", "http://www.9zwz.com/", "http://www.0460.com/", "http://www.0460.com/", "https://github.com/howie6879/mlhub123", "https://github.com/anran-world/Anranawsl", "https://github.com/wanghao221/wanghao221.github.io", "https://github.com/wanghao221/wanghao221.github.io", "https://github.com/kakuiho/wsengine"]
url2 = ["https://github.com/ruanyf/weekly/blob/master/docs/issue-228.md", "https://github.com/ruanyf/weekly/blob/master/docs/issue-227.md"]
result_link = []

# 将url中的链接提取出来输出到link.txt文件中
def get_url():
    with open("link.txt", "w") as f:
        for i in url:
            f.write(i + "\n")

# 读取txt文件中的链接到url中
def read_txt():
    url = []
    with open("link.txt", "r") as f:
        for line in f.readlines():
            url.append(line)
    return url;


def get_html(url):
    for i in url:
        html_content = requests.get(i).text
        soup = BeautifulSoup(html_content, "html.parser")
        link_nodes = soup.find_all("a")
        for node in link_nodes:
            result_link.append(node.get("href"))


def url_txt(result_link):
    with open("result_link.txt", "w") as f:
        for link in result_link:
            f.write(str(link) + "\n")


# 将txt文件中开头不是http的链接过滤掉
def filter_nothttp():
    with open("result_link.txt", "r") as f:
        with open("result_link_filter.txt", "w") as f2:
            for line in f.readlines():
                if line.startswith("http"):
                    f2.write(line)



# 将txt文件中没有“com”或者“cn”的链接过滤掉
def filter_nocomcn():
    with open("result_link_filter.txt", "r") as f:
        with open("result_link_filter1.txt", "w") as f2:
            for line in f.readlines():
                if "com" in line or "cn" in line:
                    f2.write(line)


# 将txt文件中每行com后面的过滤掉
def filter_com():
    with open("result_link_filter1.txt", "r") as f:
        with open("result_link_filter2.txt", "w") as f2:
            for line in f.readlines():
                if "com" in line:
                    line = line.split("com")[0] + "com" + "\n"
                    f2.write(line)


#将txt文件中重复的链接过滤掉
def filter_repeat():
    dif = []
    with open("result_link_filter2.txt", "r") as f:
        with open("result_link_filter02.txt", "w") as f2:
            for line in f.readlines():
                if line not in dif:
                    f2.write(line)
                    dif.append(line)

def do():
    # url1 = read_txt()
    # print(url1)
    get_html(url2)
    url_txt(result_link)
    filter_nothttp()
    filter_nocomcn()
    filter_com()
    filter_repeat()

def generate_url():
    weeklyURL = []
    for i in range(1,228):
        weeklyURL.append("https://github.com/ruanyf/weekly/blob/master/docs/issue-{}.md".format(i))
    with open("weeklyURL.txt", "w") as f:
        for i in weeklyURL:
            f.write(i + "\n")
    return weeklyURL



#https://github.com/521xueweihan/HelloGitHub/blob/master/content/HelloGitHub46.md
def genearal_url2():
    HelloURL = []
    for i in range(1, 77):
        HelloURL.append("https://github.com/521xueweihan/HelloGitHub/blob/master/content/HelloGitHub{}.md".format(i))
    return HelloURL

# WeeklyURL = generate_url()
# HelloURL = genearal_url2()
# get_html(HelloURL)
# url_txt(result_link)
# filter_nothttp()
# filter_nocomcn()
# filter_com()
# filter_repeat()


# 将result_link_filter02.txt，result_link_filter01.txt，result_link_filter3.txt文件中的链接提取出来，放到result_link_filterall.txt文件中
def filter_all():
    with open("result_link_filter02.txt", "r") as f:
        with open("result_link_filter01.txt", "r") as f2:
            with open("result_link_filter3.txt", "r") as f3:
                with open("result_link_filterall.txt", "w") as f4:
                    for line in f.readlines():
                        f4.write(line)
                    for line in f2.readlines():
                        f4.write(line)
                    for line in f3.readlines():
                        f4.write(line)
#filter_all()
# 将result_link_filterall.txt文件中的链接去重，放到result_link_filterall1.txt文件中
def filter_repeat1():
    dif = []
    with open("result_link_filterall.txt", "r") as f:
        with open("resultLink.txt", "w") as f2:
            for line in f.readlines():
                if line not in dif:
                    f2.write(line)
                    dif.append(line)
filter_repeat1()


