import numpy as np


def generateData(k, mu, sigma, dataNum):
    """
    产生混合高斯模型的数据
    :param k: 比例系数
    :param mu: 均值
    :param sigma: 标准差
    :param dataNum:数据个数
    :return: 生成的数据
    """
    # 初始化数据
    dataArray = np.zeros(dataNum, dtype=np.float32)
    # 逐个依据概率产生数据，近似于采样生成数据
    # 选取高斯分布个数
    n = len(k)
    for i in range(dataNum):
        # 产生[0,1]之间的随机数
        rand = np.random.random()
        Sum = 0
        index = 0
        while index < n:
            Sum += k[index]
            if rand < Sum:
                dataArray[i] = np.random.normal(mu[index], sigma[index])  # 按概率生成高斯分布的随机数
                break
            else:
                index += 1
    return dataArray


def normPdf(x, mu, sigma):
    """
    计算均值为mu，标准差为sigma的正态分布函数的密度函数值
    :param x: x值
    :param mu: 均值
    :param sigma: 标准差
    :return: x处的密度函数值
    """
    return 1. / ((np.sqrt(2 * np.pi)) * sigma) * (np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))


def em(dataArray, k, mu, sigma, step):
    """
    em算法估计高斯混合模型
    :param dataNum: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step:迭代次数
    :return: em 估计迭代结束估计的参数值[k,mu,sigma]
    """
    # 高斯分布个数
    n = len(k)
    # 数据个数
    dataNum = dataArray.size
    # 初始化gama数组
    gamaArray = np.zeros((n, dataNum))
    for s in range(step):
        for i in range(n):
            for j in range(dataNum):
                Sum = sum([k[t] * normPdf(dataArray[j], mu[t], sigma[t]) for t in range(n)])
                # 对每个数据计算其后验概率 gama
                gamaArray[i][j] = k[i] * normPdf(dataArray[j], mu[i], sigma[i]) / float(Sum)
        # 更新 mu
        for i in range(n):
            mu[i] = np.sum(gamaArray[i] * dataArray) / np.sum(gamaArray[i])
        # 更新 sigma
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gamaArray[i] * (dataArray - mu[i]) ** 2) / np.sum(gamaArray[i]))
        # 更新系数k
        for i in range(n):
            k[i] = np.sum(gamaArray[i]) / dataNum

    return [k, mu, sigma]


if __name__ == '__main__':
    # 参数的准确值
    k = [0.2, 0.6, 0.2]
    mu = [3, 7, 12]
    sigma = [5, 10, 15]
    # 样本数
    dataNum = 100
    # 产生数据
    dataArray = generateData(k, mu, sigma, dataNum)

    # 参数的初始值
    # 因为em算法对于参数的初始值是较为敏感的，选取不同初始参数进行迭代计算求出的结果不同
    k0 = [0.3, 0.4, 0.3]
    mu0 = [5, 6, 10]
    sigma0 = [6, 9, 12]
    step = 20
    # 使用em算法估计参数
    k1, mu1, sigma1 = em(dataArray, k0, mu0, sigma0, step)
    # 输出参数的值
    print("参数实际值:")
    print("k:", k)
    print("mu:", mu)
    print("sigma:", sigma)
    print("参数估计值:")
    print("k1:", k1)
    print("mu1:", mu1)
    print("sigma1:", sigma1)

    # 分析：估计出的参数与真实值差别略大，原因有em算法与初始参数选取有关，选取与真实值较为近似的初始值，得到的结果便
    # 接近真实值；还有一个原因是因为样本点是按照比例系数（概率）采样得到，且样本点数量较少（100个），不能保证采样得到
    # 的比例系数足够接近真实的比例系数，可以通过增大样本点数目和迭代次数提高分类准确性
