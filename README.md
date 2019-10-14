# Data-Mining-homework
Data Mining homework
# Data Mining实验报告（一）
##### 孙玉璐
---


## **实验名称：**

### Clustering with sklearn**

## **实验目的：**

###测试scikitlearn聚类算法在sklearn.dataset.load_digits()和sklearn.dataset.fetch_20newsgroups两个数据集上的聚类效果，利用NMI、homogeneity（同质性）、completeness（完整性）三个评估系数对各种聚类方法进行评估。
---


## **实验环境：**

### Windows10+Anaconda3  IDE:visual studio 2017。
---

## **实验内容步骤：**



### 1. 分别对load_digits()、fetch_20 newsgroups两个数据集进行数据初始化。


### 2. 定义一个函数bench包含算法、算法名称、数据三个参数，用于执行算法和评估指标,由于GaussianMixture算法不返回lables_值，增加了判断语句。


```python
def bench(estimator, name, data):

    t0 = time()

    estimator.fit(data)

    if hasattr(estimator,'labels_'):

           labels_pred = estimator.labels_.astype(np.int)

    else:

           labels_pred  = estimator.predict(data)

    print('%-25s\t%.3fs\t%.3f\t%.3f\t%.3f\t%.3f'

          % (name, (time() - t0), 

             metrics.normalized_mutual_info_score(labels, labels_pred,average_method='geometric'),

             metrics.homogeneity_score(labels, labels_pred),

             metrics.completeness_score(labels, labels_pred),
```


### 3. cluster各种聚类算法调用定义好的函数bench，输出实验结果。
* ##### **完整代码见目录下的digits_dataset.py和20 newsgroups_dataset.py文件**

---

## **实验数据和计算结果：**


* ### 在sklearn.dataset.load_digits()数据集上的实验结果：


|init|time|NMI|homo|comple|
|:---:|:---:|:---:|:---:|:---:|
|k-means|0.25s|0.625|0.602|0.650|
|AffinityPropagation|4.66s|0.590|0.964|0.425|
|MeanShift|0.58s|0.018|0.009|0.270|
|SpectralClustering|0.50s|0.828|0.805|0.853|
|Ward hierarchical clustering|0.14s|0.796|0.758|0.836|
|AgglomerativeClustering|0.13s|0.014|0.007|0.238|
|DBSCAN|0.37s|0.330|0.290|0.381|
|GaussianMixture|0.31s|0.837|0.623|0.622|




* ### 在sklearn.dataset.fetch_20 newsgroups数据集上的实验结果：




|init|time|NMI|homogeneity|comple|
|:---:|:---:|:---:|:---:|:---:|
|k-means|5.39s|0.417|0.364|0.488|
|AffinityPropagation|51.98s|0.289|1.000|0.169|
|MeanShift|180.81s|0.000|0.000|1.000|
|SpectralClustering|8.17s|0.666|0.656|0.677|
|Ward hierarchical clustering|55.50s|0.003|0.001|0.076|
|AgglomerativeClustering|55.47s|0.556|0.517|0.601|
|DBSCAN|0.45s|0.282|0.564|0.188|
|GaussianMixtures|111.62s|0.932|0.954|0.852|


---

## **实验总结：**

### 本次实验在两个数据集上对8种聚类方法进行了测试。通过对比可以看出在20newsgroups数据集上运行的时间复杂度较高。由结果分析，GaussianMixtures方法的NMI最大，其次是SpectralClustering。MeanShift聚类方法由于参数设置不佳致使聚类结果出现了NMI和homogeneity均为0的情况。Ward和AgglomerativeClustering两个聚类方法采用了同一种算法，使用了不同的linkage参数值，ward可作为AgglomerativeClustering的一个参数。



















