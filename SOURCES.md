# 参考来源与实现差异

本项目在 AI 辅助下按公开方法重新组织实现，没有把官方完整示例文件直接复制进仓库。算法属于常见教学方法，不主张原创算法。

## 1. 手写数字分类

- [scikit-learn 官方数字分类示例](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)：参考 8×8 digits 和 SVC 的基本流程。
- 本项目改为分层随机 75/25 划分、固定 seed=42，加入多数类基线、Macro-F1、输入校验、测试样本浏览与网页画板。
- 数据由 `load_digits()` 提供，来源与说明见 [官方 API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)。不是 28×28 MNIST。

## 2. 数值特征分类

- [scikit-learn 官方入门教程](https://scikit-learn.org/stable/getting_started.html)：参考标准化与逻辑回归组成 Pipeline 的流程。
- [load_wine 官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)：178 个样本、13 个数值特征、3 类；内置数据来自 UCI Wine。
- [UCI Wine 原始来源](https://archive.ics.uci.edu/dataset/109/wine)：数据创建者为 Stefan Aeberhard、M. Forina；数据库的许可按该页面说明适用，不由本仓库 MIT 许可证覆盖。
- 本项目加入互斥的训练／测试索引、训练部分五折交叉验证、基线、样本编辑、类别分数与特征线性贡献展示。标准化均值和方差只从训练数据计算。

## 3. Markdown 笔记检索

- [scikit-learn 文本特征提取说明](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)：参考 TF-IDF 和字符 n-gram。
- [余弦相似度官方 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)：用于检索排序。
- Markdown 切分、文件／标题／行号引用、空查询处理、网页与接口由本项目实现。演示笔记和查询为自编教学样例。
- [Sentence Transformers 语义检索说明](https://sbert.net/examples/sentence_transformer/applications/semantic-search/)仅用于后续升级学习，当前实现没有调用该库或大模型。

## 依赖与署名

- scikit-learn：BSD-3-Clause；NumPy：BSD-3-Clause。实际依赖许可以安装发行包为准。
- 本仓库代码：MIT。
- AI 使用：初版实现、文档与验证由 Codex 辅助完成；学习者应运行、理解并修改后，再对具体技术熟练程度作个人陈述。
