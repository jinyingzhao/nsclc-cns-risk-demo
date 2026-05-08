# -*- coding: utf-8 -*-
"""
NSCLC-CNS RiskMap Demo v2.0
A/C-model + C-model triple output + A-C linkage for Streamlit.

Important:
- This is a research/demo prototype for expert discussion and project presentation.
- It is NOT a clinical diagnostic or treatment decision tool.
- No SEER patient-level raw data are included.
- The interactive calculator uses transparent scoring logic calibrated to the project's model outputs
  and risk tiers for demonstration. To deploy the trained estimator directly, place the serialized
  model files under /models and replace the scoring functions accordingly.
"""

import math
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"

st.set_page_config(page_title="NSCLC-CNS RiskMap Demo", page_icon="🧠", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def risk_level_a(p: float):
    if p < 0.05:
        return "低危", "当前输入特征提示诊断时脑转移风险较低。"
    if p < 0.08:
        return "中危", "建议结合症状、分期及临床判断，关注CNS风险。"
    if p < 0.15:
        return "高危", "建议优先完善CNS影像评估，尤其是脑MRI。"
    return "极高危", "提示诊断时伴脑转移风险较高，建议重点进行CNS评估。"

def risk_level_c(p: float):
    if p < 0.50:
        return "相对低危", "在脑转移患者中属于相对较低预后风险，但仍需规范随访。"
    if p < 0.70:
        return "中危", "提示存在较高死亡风险，建议加强规范治疗与随访管理。"
    if p < 0.85:
        return "高危", "提示较高短期/中期死亡风险，建议加强MDT和综合管理。"
    return "极高危", "提示死亡风险极高，建议重点关注、密切随访与综合支持管理。"

def color_for_level(label: str):
    return {
        "低危": "#2EAD4E",
        "相对低危": "#2EAD4E",
        "中危": "#C98900",
        "高危": "#D96B00",
        "极高危": "#C62828",
    }.get(label, "#666666")

def badge(label: str):
    st.markdown(
        f"""
        <div style="display:inline-block;padding:8px 16px;border-radius:999px;
                    background:{color_for_level(label)};color:white;font-weight:700;font-size:18px;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(title, value, level, desc):
    color = color_for_level(level)
    st.markdown(
        f"""
        <div style="border:1px solid #E6EAF0;border-radius:16px;padding:18px 18px;margin-bottom:10px;background:white;box-shadow:0 1px 6px rgba(0,0,0,0.04);">
          <div style="font-size:16px;color:#475569;font-weight:700;">{title}</div>
          <div style="font-size:36px;font-weight:800;margin:8px 0;color:#0f172a;">{value}</div>
          <span style="background:{color};color:white;border-radius:999px;padding:5px 12px;font-size:14px;font-weight:700;">{level}</span>
          <div style="font-size:14px;color:#475569;margin-top:10px;line-height:1.5;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def feature_table(items):
    if items:
        st.dataframe(pd.DataFrame(items, columns=["风险贡献因素", "解释"]), use_container_width=True, hide_index=True)

# -----------------------------
# Dictionaries
# -----------------------------
histology_options = {
    "腺癌": {"a": 0.20, "c": 0.22, "explain": "腺癌在肺癌脑转移研究中常被认为具有较高CNS相关性。"},
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

sex_options = ["男", "女"]
site_options = ["上叶", "中叶", "下叶", "主支气管", "肺NOS/其他"]

# Fallback linkage data if csv missing
fallback_linkage = pd.DataFrame([
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

def load_linkage():
    p = DATA_DIR / "AC_variable_linkage_table.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            # Normalize common possible column names
            rename = {}
            for c in df.columns:
                lc = c.lower()
                if lc in ["feature", "feature_name", "variable"]: rename[c] = "变量"
                if "a" in lc and "score" in lc: rename[c] = "A_score"
                if "c" in lc and "score" in lc: rename[c] = "C_score"
            df = df.rename(columns=rename)
            if {"变量", "A_score", "C_score"}.issubset(df.columns):
                if "解释类型" not in df.columns: df["解释类型"] = "变量贡献"
                return df[["变量", "A_score", "C_score", "解释类型"]]
        except Exception:
            pass
    return fallback_linkage[["变量", "A_score", "C_score", "解释类型"]]

# -----------------------------
# Scoring functions
# -----------------------------
def calc_a(age, sex, histology, grade, tumor_size, site, bone, liver, lung, symptoms):
    score = -2.65
    score += (age - 65) / 25 * 0.25
    score += 0.06 if sex == "女" else 0.0
    score += histology_options[histology]["a"]
    score += grade_options[grade]["a"]
    score += min(tumor_size, 120) / 120 * 0.55
    score += 0.08 if site == "上叶" else 0.0
    score += 0.95 if bone else 0.0
    score += 0.38 if liver else 0.0
    score += 0.32 if lung else 0.0
    score += 0.55 if symptoms else 0.0
    return sigmoid(score)

def base_c_score(age, sex, histology, grade, tumor_size, liver, bone, lung, multi, brain_symptoms, ecog):
    score = 0.0
    score += (age - 65) / 25 * 0.80
    score += 0.12 if sex == "男" else 0.0
    score += histology_options[histology]["c"]
    score += grade_options[grade]["c"]
    score += min(tumor_size, 120) / 120 * 0.30
    score += 0.65 if liver else 0.0
    score += 0.20 if bone else 0.0
    score += 0.15 if lung else 0.0
    score += 0.35 if multi else 0.0
    score += 0.20 if brain_symptoms else 0.0
    if ecog == "2分":
        score += 0.25
    elif ecog == "3-4分":
        score += 0.65
    return score

def calc_c_triple(age, sex, histology, grade, tumor_size, liver, bone, lung, multi, brain_symptoms, ecog):
    s = base_c_score(age, sex, histology, grade, tumor_size, liver, bone, lung, multi, brain_symptoms, ecog)
    # Calibrated to project-level event rates among brain-metastasis cohort:
    # 6m all-cause death ~59.6%; 12m all-cause death ~75.0%; 12m cancer-specific death ~72.5%.
    c1_6m_all = sigmoid(0.32 + 0.88 * s)
    c2_12m_all = sigmoid(1.02 + 0.92 * s)
    c3_12m_cancer = sigmoid(0.86 + 0.90 * s)
    return c1_6m_all, c2_12m_all, c3_12m_cancer

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("NSCLC-CNS RiskMap")
st.sidebar.caption("脑转移病房｜科研Demo｜专家共建")
st.sidebar.markdown("---")
st.sidebar.warning("仅用于科研演示与专家讨论，不作为临床诊断、治疗或随访决策依据。")

# -----------------------------
# Header
# -----------------------------
st.title("NSCLC-CNS RiskMap Demo v2.0")
st.markdown("""
**定位：** 用于展示 NSCLC 患者 CNS 风险识别、脑转移后预后分层与 A-C 变量联动的科研原型工具。  
**当前升级：** C模型已更新为三联版：**6个月全因死亡风险、12个月全因死亡风险、12个月肿瘤特异性死亡风险**。
""")

st.info("数据来源说明：本项目基于 SEER 公开登记数据完成原型建模与结果校准。Demo 包不包含SEER原始患者级数据；交互计算用于展示模型逻辑与病房管理场景。")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "项目总览",
    "A模型：初诊脑转移风险",
    "C模型三联版：预后风险",
    "A-C变量联动",
    "图示解释",
    "部署与声明"
])

with tab0:
    st.header("从单点模型到脑转移病房管理工具")
    cols = st.columns(3)
    with cols[0]:
        st.subheader("A模型")
        st.markdown("**初诊时：谁可能已经伴脑转移？**  \n用于脑MRI优先级与CNS风险初筛。")
    with cols[1]:
        st.subheader("B模型")
        st.markdown("**随访中：谁未来更容易新发脑转移？**  \n当前需要真实世界随访和连续影像数据升级。")
    with cols[2]:
        st.subheader("C模型")
        st.markdown("**脑转移后：谁预后更差？**  \n当前三联输出用于短中期预后分层。")
    st.markdown("---")
    st.markdown("""
    ### 老板/专家汇报时的一句话
    > A模型解决“早识别”，B模型解决“早预警”，C模型解决“早分层”。三者未来联动后，可形成 NSCLC CNS 风险从初诊、随访到脑转移后管理的连续评估体系。
    """)

with tab1:
    st.header("A模型：诊断时脑转移风险评估")
    st.caption("适用对象：初诊 NSCLC 患者。回答：该患者诊断时伴CNS/脑转移风险是否较高？")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("年龄", 30, 90, 65, key="a_age")
        sex = st.selectbox("性别", sex_options, key="a_sex")
        histology = st.selectbox("组织学类型", list(histology_options.keys()), key="a_hist")
        grade = st.selectbox("病理分级", list(grade_options.keys()), key="a_grade")
    with c2:
        tumor_size = st.slider("肿瘤大小（mm）", 1, 150, 45, key="a_size")
        site = st.selectbox("原发部位", site_options, key="a_site")
        laterality = st.selectbox("左右侧", ["右侧", "左侧", "双侧/其他", "未知"], key="a_lat")
    with c3:
        bone_met = st.checkbox("诊断时骨转移", key="a_bone")
        liver_met = st.checkbox("诊断时肝转移", key="a_liver")
        lung_met = st.checkbox("诊断时肺内转移/对侧肺转移", key="a_lung")
        symptoms = st.checkbox("存在神经系统症状/头痛/头晕/神经功能异常", key="a_symp")

    prob = calc_a(age, sex, histology, grade, tumor_size, site, bone_met, liver_met, lung_met, symptoms)
    level, advice = risk_level_a(prob)
    st.markdown("---")
    r1, r2, r3 = st.columns([1.2, 0.8, 2])
    with r1: st.metric("预测诊断时脑转移风险", f"{prob*100:.1f}%")
    with r2:
        st.write("风险等级")
        badge(level)
    with r3: st.info(advice)

    explanations = []
    if tumor_size >= 50: explanations.append(("肿瘤大小", "原发肿瘤负荷较高，是A模型中的重要发生风险变量之一。"))
    if bone_met: explanations.append(("骨转移", "提示全身播散负荷，明显增加诊断时伴脑转移风险。"))
    if liver_met: explanations.append(("肝转移", "提示远处转移负荷增加，与CNS风险和不良预后均有关。"))
    if lung_met: explanations.append(("肺内转移/对侧肺转移", "提示疾病播散程度增加，对发生风险有一定贡献。"))
    explanations.append(("组织学类型", histology_options[histology]["explain"]))
    if grade in ["III级/低分化", "IV级/未分化"]: explanations.append(("病理分级", "较差分化通常提示肿瘤生物学行为更具侵袭性。"))
    if symptoms: explanations.append(("神经系统症状", "症状不是SEER核心建模变量，但真实临床中应直接提示完善CNS影像。"))
    st.subheader("主要风险解释")
    feature_table(explanations)

with tab2:
    st.header("C模型三联版：脑转移后短中期预后风险评估")
    st.caption("适用对象：已经确认伴脑转移的 NSCLC 患者。回答：短期/中期死亡风险是否较高？")
    st.info("当前Demo将C模型拆分为三项输出：C1=6个月全因死亡；C2=12个月全因死亡；C3=12个月肿瘤特异性死亡。")
    c1, c2, c3 = st.columns(3)
    with c1:
        c_age = st.slider("年龄", 30, 90, 68, key="c_age")
        c_sex = st.selectbox("性别", sex_options, key="c_sex")
        c_histology = st.selectbox("组织学类型", list(histology_options.keys()), key="c_hist")
        c_grade = st.selectbox("病理分级", list(grade_options.keys()), key="c_grade")
    with c2:
        c_tumor_size = st.slider("肿瘤大小（mm）", 1, 150, 55, key="c_size")
        c_ecog = st.selectbox("体能状态（真实世界升级字段）", ["0-1分", "2分", "3-4分", "未知"], key="c_ecog")
        c_brain_symptoms = st.checkbox("脑转移相关症状明显", key="c_symp")
    with c3:
        c_bone_met = st.checkbox("合并骨转移", key="c_bone")
        c_liver_met = st.checkbox("合并肝转移", key="c_liver")
        c_lung_met = st.checkbox("合并肺内转移/对侧肺转移", key="c_lung")
        c_multiple_sites = st.checkbox("多器官转移负荷明显", key="c_multi")

    p6, p12, pcs12 = calc_c_triple(c_age, c_sex, c_histology, c_grade, c_tumor_size, c_liver_met, c_bone_met, c_lung_met, c_multiple_sites, c_brain_symptoms, c_ecog)
    l6, d6 = risk_level_c(p6)
    l12, d12 = risk_level_c(p12)
    lcs, dcs = risk_level_c(pcs12)
    st.markdown("---")
    out1, out2, out3 = st.columns(3)
    with out1: metric_card("C1：6个月全因死亡风险", f"{p6*100:.1f}%", l6, d6)
    with out2: metric_card("C2：12个月全因死亡风险", f"{p12*100:.1f}%", l12, d12)
    with out3: metric_card("C3：12个月肿瘤特异性死亡风险", f"{pcs12*100:.1f}%", lcs, dcs)

    st.subheader("三联风险对比")
    bar_df = pd.DataFrame({"风险终点": ["6个月全因死亡", "12个月全因死亡", "12个月肿瘤特异性死亡"], "预测风险": [p6*100, p12*100, pcs12*100]})
    fig = px.bar(bar_df, x="风险终点", y="预测风险", text=bar_df["预测风险"].map(lambda x: f"{x:.1f}%"), range_y=[0,100], height=360)
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="预测风险（%）", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    c_expl = [("年龄", "年龄是C模型的重要预后相关变量，可能反映治疗耐受性、合并症和整体健康状态。"),
              ("组织学类型", "组织学类型在A模型和C模型中均具有较高贡献，提示其可能同时关联CNS转移倾向和预后。")]
    if c_liver_met: c_expl.append(("肝转移", "肝转移是C模型中的重要预后风险信号，提示全身疾病负荷和短期死亡风险更高。"))
    if c_tumor_size >= 50: c_expl.append(("肿瘤大小", "原发肿瘤负荷较高，对脑转移后预后也有一定解释价值。"))
    if c_multiple_sites: c_expl.append(("多器官转移", "多器官转移提示疾病负荷更重，需重点评估治疗目标和随访策略。"))
    if c_ecog in ["2分", "3-4分"]: c_expl.append(("体能状态", "体能状态不是当前SEER核心字段，但真实世界模型升级时应重点纳入。"))
    st.subheader("主要风险解释")
    feature_table(c_expl)
    st.warning("C模型用于预后风险分层展示，不用于个体患者生存期预测或治疗方案选择。")

with tab3:
    st.header("A-C变量联动：脑转移发生风险 × 脑转移后死亡风险")
    linkage = load_linkage()
    c_endpoint = st.selectbox("选择纵轴代表的C模型终点", ["12个月全因死亡风险（当前联动主图）", "6个月全因死亡风险（示意）", "12个月肿瘤特异性死亡风险（示意）"])
    plot_df = linkage.copy()
    if "6个月" in c_endpoint:
        plot_df["C_plot_score"] = (plot_df["C_score"] * 0.88 + 5).clip(0, 100)
    elif "肿瘤特异" in c_endpoint:
        plot_df["C_plot_score"] = (plot_df["C_score"] * 0.95 + 2).clip(0, 100)
    else:
        plot_df["C_plot_score"] = plot_df["C_score"]

    fig = px.scatter(plot_df, x="A_score", y="C_plot_score", text="变量", color="解释类型", size=[24]*len(plot_df),
                     hover_data=["变量", "A_score", "C_plot_score", "解释类型"],
                     labels={"A_score":"A模型贡献：诊断时脑转移风险", "C_plot_score":"C模型贡献：脑转移后预后风险"}, height=620)
    fig.update_traces(textposition="top center")
    fig.update_layout(title="A-C Linkage Map", xaxis_range=[-5,110], yaxis_range=[-5,110], legend_title_text="变量类型")
    fig.add_vline(x=plot_df["A_score"].median(), line_dash="dash", line_color="gray")
    fig.add_hline(y=plot_df["C_plot_score"].median(), line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(plot_df[["变量", "A_score", "C_plot_score", "解释类型"]].rename(columns={"C_plot_score":"C_score_selected"}), use_container_width=True, hide_index=True)

    st.markdown("""
    **解读方式：**  
    - 右上象限：既提示脑转移发生风险，也提示脑转移后预后风险，是最值得重点关注的变量。  
    - 右下象限：更偏向提示“容易发生脑转移”。  
    - 左上象限：更偏向提示“一旦发生脑转移后预后更差”。  
    - 左下象限：当前模型中综合贡献有限。
    """)

with tab4:
    st.header("图示解释")
    imgs = [
        ("HistGradientBoosting计算过程", ASSET_DIR / "machine_learning_for_risk_prediction_flow.png"),
        ("三类机器学习模型对比", ASSET_DIR / "machine_learning_models_explained_visually.png"),
        ("模型A：初诊脑转移风险", ASSET_DIR / "brain_metastasis_risk_model_overview.png"),
        ("模型B：新发脑转移随访预测", ASSET_DIR / "brain_metastasis_prediction_model_infographic.png"),
        ("模型C：脑转移后预后风险", ASSET_DIR / "brain_metastasis_prognosis_risk_model_diagram.png"),
    ]
    for title, path in imgs:
        if path.exists():
            with st.expander(title, expanded=False):
                st.image(str(path), use_container_width=True)

with tab5:
    st.header("部署与合规声明")
    st.markdown("""
    ### 数据引用建议
    Data source: Surveillance, Epidemiology, and End Results (SEER) Program, SEER*Stat Database: Incidence - SEER Research Data, 17 Registries, Nov 2025 Sub (2000–2023), released April 2026.

    ### 当前Demo的定位
    - 用于专家共建讨论、老板汇报、脑转移病房项目场景演示；
    - 不是医疗器械软件；
    - 不是临床诊断、治疗或随访决策工具；
    - 不替代脑MRI、医生判断或MDT讨论。

    ### 后续升级方向
    1. 接入中国真实世界数据；  
    2. 加入EGFR/ALK/KRAS等分子检测；  
    3. 加入脑MRI随访、新发脑转移日期、CNS-PFS；  
    4. 将B模型升级为纵向随访预测模型；  
    5. 将C模型接入治疗路径、脑转移数量/大小、脑膜转移、ECOG等变量。
    """)
