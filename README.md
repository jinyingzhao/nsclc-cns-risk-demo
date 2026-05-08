# NSCLC-CNS RiskMap Demo v2.0

这是一个用于专家沟通、老板汇报和“脑转移病房”项目讨论的 Streamlit 科研演示 Demo。

## 功能

- A模型：初诊 NSCLC 患者诊断时脑转移风险评估
- C模型三联版：
  - C1：6个月全因死亡风险
  - C2：12个月全因死亡风险
  - C3：12个月肿瘤特异性死亡风险
- A-C变量联动图：脑转移发生风险贡献 × 脑转移后预后风险贡献
- 图示解释：机器学习算法、A/B/C模型逻辑图
- 部署与数据来源说明

## Windows 本地运行

1. 解压本压缩包
2. 进入文件夹 `NSCLC_CNS_Risk_Demo_GitHub_v2`
3. 双击 `RUN_DEMO.bat`
4. 浏览器会自动打开 Demo 页面

也可以使用命令行：

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 上传 GitHub

```bash
git init
git add .
git commit -m "Initial commit: NSCLC-CNS RiskMap Demo v2"
git branch -M main
git remote add origin https://github.com/<your-name>/<repo-name>.git
git push -u origin main
```

## 上传 Streamlit Community Cloud

1. 将本文件夹上传到 GitHub 仓库
2. 打开 https://share.streamlit.io/ 或 Streamlit Community Cloud
3. 选择你的 GitHub 仓库
4. Main file path 填写：`app.py`
5. 点击 Deploy

## 数据来源说明

Data source: Surveillance, Epidemiology, and End Results (SEER) Program, SEER\*Stat Database: Incidence - SEER Research Data, 17 Registries, Nov 2025 Sub (2000–2023), released April 2026.

## 重要声明

本 Demo 仅用于科研演示、专家共建讨论和项目汇报，不作为临床诊断、治疗、随访或个体患者管理决策依据。不替代脑MRI、医生判断或MDT讨论。

当前交互计算器采用基于项目模型输出结果和风险层级校准的透明评分逻辑，用于展示工具形态与管理场景；如需部署训练后的正式模型，需要进一步接入已序列化的模型文件、中国真实世界外部验证数据及完整合规流程。
