# AI Mini Lab

三个能在普通电脑上运行的 AI 入门项目：手写数字分类、数值特征分类、带来源的 Markdown 笔记检索。适合已有 Python 和基础机器学习课程经验、想练习完整应用流程的学生。

这是 **AI 辅助复刻与学习作品**。初版代码、测试和文档由 Codex 辅助搭建，参考公开教程；它记录可运行的实现，不代表作者已经独立掌握所有算法或完成原创研究。参考与实现差异见 [SOURCES.md](SOURCES.md)，逐步学习与讲解准备见 [学习与讲解指南](docs/LEARNING.md)。

## 三个项目

| 项目 | 难度 | 做了什么 | 重点 |
|---|---|---|---|
| 手写数字识别 | 入门 | SVM 对 8×8 灰度数字分类，查看测试样本、编辑画板 | 训练与测试划分、准确率、Macro-F1、混淆矩阵 |
| 数值特征分类 | 入门进阶 | UCI Wine 13 特征 + StandardScaler + 逻辑回归，展示分数与线性特征贡献 | Pipeline 避免数据泄漏、训练集交叉验证、基线比较 |
| 笔记检索 | 略高于基础 | Markdown 按标题和段落切分，字符 TF-IDF + 余弦相似度，显示文件、标题与行号 | 文本处理、检索排序、引用定位、HTTP 接口与网页联调 |

三个项目安装依赖后无需网络或 API Key。检索项目使用自行编写的示例笔记；仓库没有个人笔记、简历、聊天记录或联系方式。

## 快速运行

推荐 Python 3.13，验证环境为 Python 3.13.7、scikit-learn 1.9.0、NumPy 2.4.4。

```bash
git clone https://github.com/RI-MuShengHan/ai-mini-lab.git
cd ai-mini-lab
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python -m mini_lab.evaluate all
python -m unittest discover -s tests -v
python -m mini_lab.web --port 8765
```

打开 <http://127.0.0.1:8765>。数字页点击“下一个测试样本”；特征页加载样本再修改某项数值；检索页输入“任务中断了如何避免重复生成”。按 Ctrl+C 停止服务。若 PowerShell 无法激活环境，可直接用 `.venv\Scripts\python.exe` 替代上面的 `python`。

## 验证结果

2026-10-05 在上述环境运行，参数事先固定，未用测试集挑选参数。

| 项目 | 划分／样例 | 结果 | 对照 |
|---|---|---|---|
| digits | 分层 75/25，seed=42，1347 训练 / 450 测试 | Accuracy 98.89%，Macro-F1 0.9888 | 多数类基线 10.22% |
| Wine | 分层 75/25，seed=42，133 训练 / 45 测试 | Accuracy 100%，Macro-F1 1.0000 | 多数类基线 40.00%；仅训练部分的五折 Macro-F1 均值 0.9777 |
| 检索 | 2 份自编笔记，8 个自编查询 | Hit@1 8/8，Hit@3 8/8 | 小规模功能检查，不能当成外部泛化评测 |

13 项自动化检查通过，覆盖划分互斥、标准化仅拟合训练集、输入校验、引用原文行号、段落边界及接口行为。完整指标和混淆矩阵保存在 [artifacts](artifacts)。

**结果边界：** digits 是 sklearn 内置 8×8 数据，随手涂画可能偏离其分布；Wine 测试集只有 45 个样本，100% 不代表真实业务效果。检索是词面匹配，暂未接入向量嵌入或大模型回答。分类分数未经概率校准，特征贡献也不是因果解释。

## 目录

```text
mini_lab/       算法、评估 CLI、本地 HTTP 服务
web/            三个实验的网页界面
data/           自编示例笔记与检索查询
tests/          自动化检查
artifacts/      实际运行的指标 JSON
docs/           学习、演示和讲解指南
```

服务绑定本机回环地址，适合本地演示。要扩展成对外服务，需要另行考虑鉴权、并发与部署。替换自己的笔记时，可在本地修改 `web.py` 中 `note_index()` 的路径，并将笔记保存在已忽略的 `private_notes/` 目录。

## 可以自己完成的下一步

1. 找出一个数字误判样本，描述图像与预测的差异。
2. 在训练部分比较不同模型，保留测试集作最后一次评估。
3. 加入一份自己的示例笔记，再补充未用来设计检索器的查询。
4. 在单独分支尝试语义检索，比较同义词查询与字符 TF-IDF 的差异。

代码采用 MIT 许可证；第三方库与数据集保留各自的许可和来源说明。
