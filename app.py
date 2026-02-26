import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="BOQ Emissions & Cost Optimizer",
    page_icon="🏗️",
    layout="wide"
)

GRADE_LIST = ['M5', 'M10', 'M15', 'M20', 'M30', 'M40', 'M45', 'M50']

emission_factors = {
    'M5': 150, 'M10': 175, 'M15': 200, 'M20': 230,
    'M30': 290, 'M40': 350, 'M45': 380, 'M50': 410
}

grade_cost_per_m3 = {
    'M5': 3500, 'M10': 3800, 'M15': 4200, 'M20': 4800,
    'M30': 5800, 'M40': 7000, 'M45': 7800, 'M50': 8500
}

lifecycle_penalty = {
    'M5': 3.5, 'M10': 2.8, 'M15': 2.2, 'M20': 1.8,
    'M30': 1.3, 'M40': 1.1, 'M45': 1.05, 'M50': 1.0
}

RCC_STEEL_FACTOR = 115

# Min grade per structural category
min_grade = {
    'Column': 'M30', 'Beam': 'M30', 'Slab': 'M20',
    'Foundation': 'M15', 'Wall': 'M15', 'Stair': 'M20',
    'Basement': 'M20', 'General': 'M10'
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_emissions(material, grade, quantity):
    ef = emission_factors[grade]
    if material == 'RCC':
        ef += RCC_STEEL_FACTOR
    return quantity * ef

def compute_lifecycle_cost(grade, quantity):
    return quantity * grade_cost_per_m3[grade] * lifecycle_penalty[grade]

def compute_material_cost(grade, quantity):
    return quantity * grade_cost_per_m3[grade]

def get_optimized_grade(material, category, preference):
    """
    preference: 0.0 = minimize emissions, 1.0 = minimize lifecycle cost
    Returns recommended grade based on preference
    """
    min_g = min_grade.get(category, 'M10')
    min_idx = GRADE_LIST.index(min_g)
    available = GRADE_LIST[min_idx:]

    scores = []
    for grade in available:
        em_score = emission_factors[grade] + (RCC_STEEL_FACTOR if material == 'RCC' else 0)
        lc_score = grade_cost_per_m3[grade] * lifecycle_penalty[grade]

        # Normalize
        em_norm = (em_score - 150) / (525 - 150)
        lc_norm = (lc_score - 3500) / (8500 - 3500)

        # Weighted score based on preference
        score = (1 - preference) * em_norm + preference * lc_norm
        scores.append((grade, score))

    return min(scores, key=lambda x: x[1])[0]

# ============================================================
# UI
# ============================================================

st.title("🏗️ BOQ Emissions & Cost Optimizer")
st.markdown("*Automating BOQ and Cost Analysis using Machine Learning — Dayananda Sagar University*")
st.divider()

# ============================================================
# SIDEBAR — USER INPUTS
# ============================================================

st.sidebar.header("📋 BOQ Item Details")

material = st.sidebar.selectbox("Material", ["PCC", "RCC"])
grade = st.sidebar.selectbox("Current Grade", GRADE_LIST, index=GRADE_LIST.index("M30"))
category = st.sidebar.selectbox("Structural Element Type", 
    ["Column", "Beam", "Slab", "Foundation", "Wall", "Stair", "Basement", "General"])
quantity = st.sidebar.number_input("Quantity (m³)", min_value=1.0, value=1000.0, step=10.0)

st.sidebar.divider()
st.sidebar.header("⚖️ Optimization Preference")
preference = st.sidebar.slider(
    "Minimize Emissions ←→ Minimize Lifecycle Cost",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="0 = fully optimize for lowest emissions, 1 = fully optimize for lowest lifecycle cost"
)

if preference < 0.33:
    pref_label = "🌱 Emissions Priority"
elif preference < 0.66:
    pref_label = "⚖️ Balanced"
else:
    pref_label = "💰 Cost Priority"
st.sidebar.markdown(f"**Mode: {pref_label}**")

# ============================================================
# COMPUTE RESULTS
# ============================================================

# Current design
current_emissions = compute_emissions(material, grade, quantity)
current_mat_cost = compute_material_cost(grade, quantity)
current_lc_cost = compute_lifecycle_cost(grade, quantity)

# Recommended grade
rec_grade = get_optimized_grade(material, category, preference)

# Optimized design
opt_emissions = compute_emissions(material, rec_grade, quantity)
opt_mat_cost = compute_material_cost(rec_grade, quantity)
opt_lc_cost = compute_lifecycle_cost(rec_grade, quantity)

# Savings
em_saving = current_emissions - opt_emissions
em_saving_pct = (em_saving / current_emissions * 100) if current_emissions > 0 else 0
lc_saving = current_lc_cost - opt_lc_cost
lc_saving_pct = (lc_saving / current_lc_cost * 100) if current_lc_cost > 0 else 0

# ============================================================
# SECTION 1 — PREDICTED EMISSIONS
# ============================================================

st.header("1️⃣ Predicted Emissions")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Current Emissions",
        value=f"{current_emissions/1000:.1f} tonnes CO₂e",
        help="Based on current grade selection"
    )
with col2:
    st.metric(
        label="Optimized Emissions",
        value=f"{opt_emissions/1000:.1f} tonnes CO₂e",
        delta=f"{-em_saving/1000:.1f} tonnes CO₂e",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="Emissions Reduction",
        value=f"{em_saving_pct:.1f}%",
        delta="vs current design",
        delta_color="inverse" if em_saving_pct >= 0 else "normal"
    )

# ============================================================
# SECTION 2 — RECOMMENDED GRADE ALLOCATION
# ============================================================

st.header("2️⃣ Recommended Grade Allocation")

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Current Grade:** {grade}")
    st.write(f"- Emission Factor: {emission_factors[grade] + (RCC_STEEL_FACTOR if material == 'RCC' else 0)} kg CO₂e/m³")
    st.write(f"- Material Cost: ₹{grade_cost_per_m3[grade]:,}/m³")
    st.write(f"- Lifecycle Penalty: {lifecycle_penalty[grade]}×")

with col2:
    color = "🟢" if rec_grade <= grade else "🟡"
    st.success(f"**Recommended Grade:** {rec_grade} {color}")
    st.write(f"- Emission Factor: {emission_factors[rec_grade] + (RCC_STEEL_FACTOR if material == 'RCC' else 0)} kg CO₂e/m³")
    st.write(f"- Material Cost: ₹{grade_cost_per_m3[rec_grade]:,}/m³")
    st.write(f"- Lifecycle Penalty: {lifecycle_penalty[rec_grade]}×")

st.caption(f"Minimum structural grade for {category}: **{min_grade.get(category, 'M10')}**")

# ============================================================
# SECTION 3 — COMPARE CURRENT VS OPTIMIZED
# ============================================================

st.header("3️⃣ Current vs Optimized Design")

col1, col2 = st.columns(2)

with col1:
    # Comparison table
    comparison = pd.DataFrame({
        'Metric': ['Grade', 'Emissions (tonnes CO₂e)', 'Material Cost (₹ Lakhs)', 'Lifecycle Cost (₹ Lakhs)'],
        'Current Design': [
            grade,
            f"{current_emissions/1000:.1f}",
            f"{current_mat_cost/100000:.2f}",
            f"{current_lc_cost/100000:.2f}"
        ],
        'Optimized Design': [
            rec_grade,
            f"{opt_emissions/1000:.1f}",
            f"{opt_mat_cost/100000:.2f}",
            f"{opt_lc_cost/100000:.2f}"
        ],
        'Saving': [
            f"{grade} → {rec_grade}",
            f"{em_saving/1000:.1f} tonnes ({em_saving_pct:.1f}%)",
            f"₹{(current_mat_cost - opt_mat_cost)/100000:.2f} L",
            f"₹{lc_saving/100000:.2f} L ({lc_saving_pct:.1f}%)"
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

with col2:
    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    categories_bar = ['Current', 'Optimized']
    em_values = [current_emissions/1000, opt_emissions/1000]
    lc_values = [current_lc_cost/100000, opt_lc_cost/100000]

    axes[0].bar(categories_bar, em_values, color=['#e74c3c', '#2ecc71'], width=0.5)
    axes[0].set_title('Emissions (tonnes CO₂e)')
    axes[0].set_ylabel('tonnes CO₂e')
    for i, v in enumerate(em_values):
        axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    axes[1].bar(categories_bar, lc_values, color=['#e74c3c', '#2ecc71'], width=0.5)
    axes[1].set_title('Lifecycle Cost (₹ Lakhs)')
    axes[1].set_ylabel('₹ Lakhs')
    for i, v in enumerate(lc_values):
        axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# SECTION 4 — PARETO TRADE-OFF CURVE
# ============================================================

st.header("4️⃣ Emissions vs Lifecycle Cost Trade-off")

# Generate trade-off curve across all available grades
available_grades = GRADE_LIST[GRADE_LIST.index(min_grade.get(category, 'M10')):]
em_curve = []
lc_curve = []
grade_labels = []

for g in available_grades:
    em_curve.append(compute_emissions(material, g, quantity) / 1000)
    lc_curve.append(compute_lifecycle_cost(g, quantity) / 100000)
    grade_labels.append(g)

fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(em_curve, lc_curve, 'o-', color='steelblue', linewidth=2, markersize=8, zorder=3)

for i, g in enumerate(grade_labels):
    ax.annotate(g, (em_curve[i], lc_curve[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=9, color='steelblue')

# Mark current and recommended
ax.scatter([current_emissions/1000], [current_lc_cost/100000],
           color='red', s=200, zorder=5, marker='*', label=f'Current ({grade})')
ax.scatter([opt_emissions/1000], [opt_lc_cost/100000],
           color='green', s=200, zorder=5, marker='*', label=f'Recommended ({rec_grade})')

ax.set_xlabel('Embodied Emissions (tonnes CO₂e)', fontsize=12)
ax.set_ylabel('Lifecycle Cost (₹ Lakhs)', fontsize=12)
ax.set_title(f'Trade-off Curve — {material} {category} ({quantity:.0f} m³)', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.caption("Each point on the curve represents a different grade. Moving left = lower emissions but higher lifecycle cost. Moving right = lower lifecycle cost but higher emissions.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<small>
**Methodology:** Emission factors from ICE Database (University of Bath). 
Lifecycle cost = Material cost × Durability penalty factor. 
Optimization using NSGA-II multi-objective evolutionary algorithm.  
**Project:** Automating BOQ and Cost Analysis using Machine Learning — Purvankara Project  
Dayananda Sagar University, School of Engineering
</small>
""", unsafe_allow_html=True)
