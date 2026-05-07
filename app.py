
# -*- coding: utf-8 -*-
"""
NSCLC-CNS Risk Stratification Demo v1.0

This Streamlit demo is a physician-facing prototype for project discussion.
It has three modules:
1) A model: brain metastasis risk at diagnosis
2) C model: 12-month all-cause mortality risk among patients with brain metastasis
3) A-C linkage: occurrence-risk vs post-metastasis prognosis factor map

Important:
- This demo is NOT a clinical decision tool.
- The current version uses transparent scoring weights derived from model importance results,
  not a deployed final clinical model.
- It is intended for expert discussion, internal reporting, and real-world data collaboration design.
"""

import math
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="NSCLC-CNS Risk Stratification Demo",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Helper functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def risk_level_a(p):
    if p < 0.05:
        return "低危", "当前输入特征提示诊断时脑转移风险较低。"
    elif p < 0.08:
        return "中危", "建议结合症状、分期及临床判断，关注 CNS 风险。"
    elif p < 0.15:
        return "高危", "建议优先完善 CNS 影像评估，尤其是脑 MRI。"
    else:
        return "极高危", "提示诊断时伴脑转移风险较高，建议重点进行 CNS 评估。"

def risk_level_c(p):
    if p < 0.50:
        return "相对低危", "在脑转移患者中属于相对较低死亡风险，但仍需密切随访。"
    elif p < 0.70:
        return "中危", "提示存在较高 12 个月死亡风险，建议规范治疗与随访管理。"
    elif p < 0.85:
        return "高危", "提示较高短期死亡风险，建议加强多学科管理和治疗策略评估。"
    else:
        return "极高危", "提示 12 个月死亡风险极高，建议重点关注、密切随访及综合管理。"

def badge(label):
    color_map = {
        "低危": "#2EAD4E",
        "中危": "#C98900",
        "高危": "#D96B00",
        "极高危": "#C62828",
        "相对低危": "#2EAD4E",
    }
    color = color_map.get(label, "#666666")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:8px 16px;border-radius:999px;
                    background:{color};color:white;font-weight:700;font-size:18px;">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

def feature_explanation_table(items):
    df = pd.DataFrame(items, columns=["风险贡献因素", "解释"])
    st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Data dictionaries
# -----------------------------
histology_options = {
    "腺癌": {"a": 0.20, "c": 0.22, "explain": "腺癌在肺癌脑转移研究中常被认为具有较高 CNS 相关性。"},
    "鳞癌": {"a": -0.05, "c": -0.02, "explain": "相较腺癌，脑转移倾向通常较低。"},
    "大细胞/未分化癌": {"a": 0.15, "c": 0.18, "explain": "可能提示侵袭性较强，需结合分期判断。"},
    "NSCLC NOS/其他": {"a": 0.08, "c": 0.10, "explain": "组织学信息不充分时，模型解释能力会下降。"},
}

grade_options = {
    "I级/高分化": {"a": -0.10, "c": -0.04},
    "II级/中分化": {"a": 0.00, "c": 0.00},
    "III级/低分化": {"a": 0.16, "c": 0.03},
    "IV级/未分化": {"a": 0.22, "c": 0.05},
    "未知/无法评估": {"a": 0.05, "c": 0.00},
}

site_options = ["上叶", "中叶", "下叶", "主支气管", "肺 NOS/其他"]
sex_options = ["男", "女"]

# A/C importance table based on current project outputs
linkage_data = pd.DataFrame([
    ["tumor_size", "肿瘤大小", 100.0, 24.9, "双高/偏发生风险"],
    ["bone_met", "骨转移", 71.8, 6.3, "发生风险主导"],
    ["histology", "组织学类型", 68.2, 100.0, "双高因素"],
    ["grade", "病理分级", 48.1, 0.0, "发生风险主导"],
    ["age", "年龄", 46.8, 93.7, "双高/偏预后风险"],
    ["liver_met", "肝转移", 21.1, 44.4, "预后风险主导"],
    ["lung_met", "肺内转移", 20.2, 3.0, "低-中贡献"],
    ["laterality", "左右侧", 2.3, 0.0, "低贡献"],
    ["primary_site", "原发部位", 1.9, 4.3, "低贡献"],
    ["sex", "性别", 1.4, 18.9, "预后弱相关"],
], columns=["feature_key", "变量", "A_score", "C_score", "解释类型"])

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("NSCLC-CNS Demo")
st.sidebar.caption("科研原型｜专家沟通｜内部汇报")
st.sidebar.markdown("---")
st.sidebar.info(
    "本工具当前基于 SEER 原型模型结果进行演示，"
    "用于专家讨论和真实世界研究设计，不用于临床诊断或治疗决策。"
)

# -----------------------------
# Header
# -----------------------------
st.title("NSCLC-CNS 风险分层 Demo")
st.markdown(
    """
    这是一个面向专家沟通和内部汇报的科研原型工具，用于展示：
    **诊断时脑转移风险评估（A模型）**、**脑转移后12个月死亡风险评估（C模型）**，
    以及 **A-C 变量联动分析**。
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "① A模型：诊断时脑转移风险",
    "② C模型：脑转移后12个月死亡风险",
    "③ A-C变量联动图",
    "④ 项目说明"
])

# -----------------------------
# Tab 1: A model demo
# -----------------------------
with tab1:
    st.header("A模型：诊断时脑转移风险评估")
    st.caption("回答问题：这个 NSCLC 患者在初诊时是否更可能已经伴有脑转移？")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("年龄", 30, 90, 65, key="a_age")
        sex = st.selectbox("性别", sex_options, key="a_sex")
        histology = st.selectbox("组织学类型", list(histology_options.keys()), key="a_hist")
        grade = st.selectbox("病理分级", list(grade_options.keys()), key="a_grade")

    with col2:
        tumor_size = st.slider("肿瘤大小（mm）", 1, 150, 45, key="a_size")
        site = st.selectbox("原发部位", site_options, key="a_site")
        laterality = st.selectbox("左右侧", ["右侧", "左侧", "双侧/其他", "未知"], key="a_lat")

    with col3:
        bone_met = st.checkbox("诊断时骨转移", key="a_bone")
        liver_met = st.checkbox("诊断时肝转移", key="a_liver")
        lung_met = st.checkbox("诊断时肺内转移/对侧肺转移", key="a_lung")
        symptoms = st.checkbox("存在神经系统症状/头痛/头晕/神经功能异常", key="a_symp")

    # Transparent scoring: calibrated to plausible distribution around 2-30%
    score = -2.65
    score += (age - 65) / 25 * 0.25
    score += 0.06 if sex == "女" else 0.0
    score += histology_options[histology]["a"]
    score += grade_options[grade]["a"]
    score += min(tumor_size, 120) / 120 * 0.55
    score += 0.08 if site == "上叶" else 0.0
    score += 0.95 if bone_met else 0.0
    score += 0.38 if liver_met else 0.0
    score += 0.32 if lung_met else 0.0
    score += 0.55 if symptoms else 0.0

    prob = sigmoid(score)
    level, advice = risk_level_a(prob)

    st.markdown("---")
    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.metric("预测脑转移风险", f"{prob*100:.1f}%")
    with r2:
        st.write("风险等级")
        badge(level)
    with r3:
        st.info(advice)

    explanations = []
    if tumor_size >= 50:
        explanations.append(("肿瘤大小", "原发肿瘤负荷较高，是当前A模型中最重要的发生风险变量之一。"))
    if bone_met:
        explanations.append(("骨转移", "提示全身播散负荷，明显增加诊断时伴脑转移风险。"))
    if liver_met:
        explanations.append(("肝转移", "提示远处转移负荷增加，与 CNS 风险和不良预后均有关。"))
    if lung_met:
        explanations.append(("肺内转移", "提示疾病播散程度增加，对发生风险有一定贡献。"))
    explanations.append(("组织学类型", histology_options[histology]["explain"]))
    if grade in ["III级/低分化", "IV级/未分化"]:
        explanations.append(("病理分级", "较差分化通常提示肿瘤生物学行为更具侵袭性。"))
    if symptoms:
        explanations.append(("神经系统症状", "症状不是SEER核心建模变量，但在真实临床中应直接提示完善CNS影像。"))

    st.subheader("主要风险解释")
    feature_explanation_table(explanations)

    st.warning("说明：当前A模型 Demo 用于科研原型展示；真实临床中脑转移诊断仍需脑MRI/增强MRI等影像学证据。")

# -----------------------------
# Tab 2: C model demo
# -----------------------------
with tab2:
    st.header("C模型：脑转移后12个月全因死亡风险评估")
    st.caption("回答问题：已经伴脑转移的 NSCLC 患者，12个月内死亡风险是否更高？")

    st.info("C模型仅适用于已经确认存在脑转移的患者。若患者尚未确认脑转移，应先使用A模型进行CNS风险评估。")

    col1, col2, col3 = st.columns(3)

    with col1:
        c_age = st.slider("年龄", 30, 90, 68, key="c_age")
        c_sex = st.selectbox("性别", sex_options, key="c_sex")
        c_histology = st.selectbox("组织学类型", list(histology_options.keys()), key="c_hist")
        c_grade = st.selectbox("病理分级", list(grade_options.keys()), key="c_grade")

    with col2:
        c_tumor_size = st.slider("肿瘤大小（mm）", 1, 150, 55, key="c_size")
        c_ecog = st.selectbox("体能状态（探索性输入）", ["0-1分", "2分", "3-4分", "未知"], key="c_ecog")
        c_brain_symptoms = st.checkbox("脑转移相关症状明显", key="c_symp")

    with col3:
        c_bone_met = st.checkbox("合并骨转移", key="c_bone")
        c_liver_met = st.checkbox("合并肝转移", key="c_liver")
        c_lung_met = st.checkbox("合并肺内转移/对侧肺转移", key="c_lung")
        c_multiple_sites = st.checkbox("多器官转移负荷明显", key="c_multi")

    # Transparent scoring calibrated around high event rates among BM patients
    c_score = 0.85
    c_score += (c_age - 65) / 25 * 0.80
    c_score += 0.12 if c_sex == "男" else 0.0
    c_score += histology_options[c_histology]["c"]
    c_score += grade_options[c_grade]["c"]
    c_score += min(c_tumor_size, 120) / 120 * 0.30
    c_score += 0.65 if c_liver_met else 0.0
    c_score += 0.20 if c_bone_met else 0.0
    c_score += 0.15 if c_lung_met else 0.0
    c_score += 0.35 if c_multiple_sites else 0.0
    c_score += 0.20 if c_brain_symptoms else 0.0
    if c_ecog == "2分":
        c_score += 0.25
    elif c_ecog == "3-4分":
        c_score += 0.65

    c_prob = sigmoid(c_score)
    c_level, c_advice = risk_level_c(c_prob)

    st.markdown("---")
    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.metric("预测12个月全因死亡风险", f"{c_prob*100:.1f}%")
    with r2:
        st.write("风险等级")
        badge(c_level)
    with r3:
        st.info(c_advice)

    c_expl = []
    c_expl.append(("年龄", "在C模型中，年龄是最重要的预后相关变量之一，可能反映治疗耐受性、合并症和整体健康状态。"))
    c_expl.append(("组织学类型", "组织学类型在A模型和C模型中均具有较高贡献，提示其可能同时关联CNS转移倾向和预后。"))
    if c_liver_met:
        c_expl.append(("肝转移", "肝转移是C模型中的重要预后风险信号，提示全身疾病负荷和短期死亡风险更高。"))
    if c_tumor_size >= 50:
        c_expl.append(("肿瘤大小", "原发肿瘤负荷较高，对脑转移后预后也有一定解释价值。"))
    if c_multiple_sites:
        c_expl.append(("多器官转移", "多器官转移提示疾病负荷更重，需重点评估治疗目标和随访策略。"))
    if c_ecog in ["2分", "3-4分"]:
        c_expl.append(("体能状态", "体能状态不是当前SEER核心字段，但真实世界模型升级时应重点纳入。"))

    st.subheader("主要风险解释")
    feature_explanation_table(c_expl)

    st.warning("说明：当前C模型用于预后分层展示，不用于个体患者生存期预测或治疗决策。")

# -----------------------------
# Tab 3: Linkage
# -----------------------------
with tab3:
    st.header("A-C变量联动：脑转移发生风险 × 脑转移后死亡风险")
    st.caption("横轴：变量对诊断时脑转移风险的贡献；纵轴：变量对脑转移后12个月死亡风险的贡献。")

    fig = px.scatter(
        linkage_data,
        x="A_score",
        y="C_score",
        text="变量",
        color="解释类型",
        size=[24]*len(linkage_data),
        hover_data=["变量", "A_score", "C_score", "解释类型"],
        labels={
            "A_score": "A模型贡献：诊断时脑转移风险",
            "C_score": "C模型贡献：12个月死亡风险"
        },
        height=620
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title="A-C Linkage Map",
        xaxis_range=[-5, 110],
        yaxis_range=[-5, 110],
        legend_title_text="变量类型"
    )
    fig.add_vline(x=linkage_data["A_score"].median(), line_dash="dash")
    fig.add_hline(y=linkage_data["C_score"].median(), line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("变量联动表")
    st.dataframe(linkage_data[["变量", "A_score", "C_score", "解释类型"]], use_container_width=True, hide_index=True)

    st.markdown(
        """
        **解读方式：**

        - 右上象限：既提示脑转移发生风险，也提示脑转移后预后风险，是最值得重点关注的变量。
        - 右下象限：更偏向提示“容易发生脑转移”。
        - 左上象限：更偏向提示“一旦发生脑转移后预后更差”。
        - 左下象限：当前模型中综合贡献有限。
        """
    )

# -----------------------------
# Tab 4: Project explanation
# -----------------------------
with tab4:
    st.header("项目说明：这个 Demo 应该怎么讲？")

    st.markdown(
        """
        ### 一句话定位

        本项目希望基于公开肿瘤登记数据和后续中国真实世界临床数据，
        建立一个面向 NSCLC 患者的 **CNS 转移风险与转移后预后风险分层工具原型**。

        ### 当前已经完成

        1. **A模型：诊断时脑转移风险模型**  
           基于 SEER 2010–2023 数据，使用初诊时临床病理变量，评估患者诊断时伴脑转移风险。

        2. **C模型：脑转移后12个月死亡风险模型**  
           在初诊伴脑转移患者中，加入 survival months、vital status 等随访字段，评估早期死亡风险。

        3. **A-C联动分析**  
           比较变量对“脑转移发生风险”和“脑转移后死亡风险”的贡献，形成连续疾病管理视角。

        ### 这个工具不是什么？

        - 不是临床诊断工具；
        - 不是替代脑MRI；
        - 不是治疗决策工具；
        - 不是已经完成临床验证的终版模型。

        ### 这个工具是什么？

        - 是一个科研原型；
        - 是专家讨论工具；
        - 是真实世界研究设计框架；
        - 是后续中国数据验证和疾病管理工具化的基础。

        ### 下一步需要专家帮助

        1. 判断变量是否符合临床逻辑；  
        2. 判断风险分层阈值是否有临床解释价值；  
        3. 帮助设计中国真实世界数据采集表；  
        4. 共同验证模型在中国 NSCLC，尤其是驱动基因阳性患者中的适用性。
        """
    )
