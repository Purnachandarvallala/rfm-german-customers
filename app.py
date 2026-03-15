# ══════════════════════════════════════════════════════
# RFM Customer Segmentation — Streamlit Dashboard
# Author: Purnachandar Vallala
# ══════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────
st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        font-size: 28px !important;
        margin: 0 !important;
        font-weight: 800 !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.8) !important;
        margin: 6px 0 0 !important;
        font-size: 14px !important;
    }
    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #1a1a2e;
        margin: 0;
    }
    .kpi-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 4px 0 0;
    }
    .insight-box {
        background: #f0fffe;
        border: 1px solid #2EC4B6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 14px;
        color: #1a1a2e;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# DATA LOADING — Run entire RFM pipeline
# ══════════════════════════════════════════════════════
@st.cache_data
def load_and_process_data():
    """
    Loads raw data and runs complete RFM pipeline.
    @st.cache_data means this only runs ONCE —
    subsequent page interactions use cached results.
    """
    # ── Load ──────────────────────────────────────────
    @st.cache_data
def load_raw_data():
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00352/"
        "Online%20Retail.xlsx"
    )
    df = pd.read_excel(url)
    df = df.rename(columns={
        'Customer ID': 'CustomerID' if 'Customer ID'
        in df.columns else 'CustomerID'
    })
    return df

    # ── Clean ─────────────────────────────────────────
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Customer ID'] = df['Customer ID'].astype(int).astype(str)
    df = df.rename(columns={'Customer ID': 'CustomerID'})
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # ── RFM Calculation ───────────────────────────────
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate',
                     lambda x: (reference_date - x.max()).days),
        Frequency = ('Invoice', 'nunique'),
        Monetary  = ('TotalPrice', 'sum')
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].round(2)

    # ── Scoring ───────────────────────────────────────
    rfm['R_Score'] = pd.qcut(
        rfm['Recency'], q=5,
        labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(
        rfm['Frequency'].rank(method='first'), q=5,
        labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(
        rfm['Monetary'].rank(method='first'), q=5,
        labels=[1,2,3,4,5]).astype(int)

    rfm['RFM_Score'] = (rfm['R_Score'].astype(str) +
                        rfm['F_Score'].astype(str) +
                        rfm['M_Score'].astype(str))
    rfm['RFM_Total'] = (rfm['R_Score'] +
                        rfm['F_Score'] +
                        rfm['M_Score'])

    # ── Segments ──────────────────────────────────────
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        elif r <= 2 and f >= 2 and m >= 2:
            return 'At Risk'
        elif r == 2 and f <= 2:
            return 'About To Sleep'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        else:
            return 'Needs Attention'

    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    # ── CLV ───────────────────────────────────────────
    rfm['CLV'] = ((rfm['Frequency'] * rfm['Monetary'])
                  / rfm['Recency']).round(2)
    rfm['CLV_Tier'] = pd.qcut(
        rfm['CLV'].rank(method='first'), q=4,
        labels=['Bronze','Silver','Gold','Platinum']
    ).astype(str)

    # ── Country mapping ───────────────────────────────
    customer_country = df.groupby('CustomerID')['Country'] \
                         .first().reset_index()
    rfm = rfm.merge(customer_country, on='CustomerID', how='left')

    return df, rfm

# ── Load data with spinner ────────────────────────────
with st.spinner('Loading and processing data...'):
    df_clean, rfm = load_and_process_data()

# ── German customers ──────────────────────────────────
german_ids = df_clean[
    df_clean['Country'] == '"🇩🇪 Germany'
]['CustomerID'].unique()
rfm_germany = rfm[rfm['CustomerID'].isin(german_ids)].copy()

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

st.sidebar.markdown("&nbsp;")
st.sidebar.title("📊 RFM Dashboard")
st.sidebar.markdown("**E-Commerce Segmentation**")
st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.radio(
    "📌 Navigate",
    ["🌍 Global Overview",
     "🏆 German Deep Dive",
     "💎 CLV Analysis",
     "📅 Cohort Retention"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# Segment filter
all_segments = sorted(rfm['Segment'].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=all_segments,
    default=all_segments
)

# Monetary filter
min_monetary = st.sidebar.slider(
    "Min. Monetary Value (£)",
    min_value=0,
    max_value=int(rfm['Monetary'].quantile(0.95)),
    value=0,
    step=50
)

# Apply filters
rfm_filtered = rfm[
    (rfm['Segment'].isin(selected_segments)) &
    (rfm['Monetary'] >= min_monetary)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Showing:** {len(rfm_filtered):,} customers"
)
st.sidebar.markdown(
    f"**Revenue:** £{rfm_filtered['Monetary'].sum():,.0f}"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by **Purnachandar Vallala**  \n"
    "MSc Data Science Student · Germany"
)

# ══════════════════════════════════════════════════════
# COLOUR MAP
# ══════════════════════════════════════════════════════
COLORS = {
    'Champions'          : '#2EC4B6',
    'Loyal Customers'    : '#3D9970',
    'Potential Loyalists': '#89B4FA',
    'New Customers'      : '#A6E3A1',
    'Needs Attention'    : '#FAB387',
    'About To Sleep'     : '#F4A261',
    'At Risk'            : '#E07A5F',
    "Can't Lose Them"    : '#F38BA8',
    'Hibernating'        : '#6C7086'
}

# ══════════════════════════════════════════════════════
# PAGE 1: GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════
if page == "🌍 Global Overview":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Global Customer Segmentation Dashboard</h1>
        <p>RFM Analysis · 5,878 Customers · 38 Countries · 
           UCI Online Retail II Dataset (2009–2011)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers",
              f"{len(rfm_filtered):,}")
    k2.metric("Total Revenue",
              f"£{rfm_filtered['Monetary'].sum():,.0f}")
    k3.metric("Champions",
              f"{(rfm_filtered['Segment']=='Champions').sum():,}",
              f"{(rfm_filtered['Segment']=='Champions').mean()*100:.1f}%")
    k4.metric("At Risk",
              f"{(rfm_filtered['Segment']=='At Risk').sum():,}",
              f"-{(rfm_filtered['Segment']=='At Risk').mean()*100:.1f}%",
              delta_color="inverse")
    k5.metric("Avg Customer Value",
              f"£{rfm_filtered['Monetary'].mean():,.0f}")

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👥 Customer Count by Segment")
        seg_counts = rfm_filtered['Segment'] \
            .value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        seg_counts['Pct'] = (
            seg_counts['Count'] /
            seg_counts['Count'].sum() * 100
        ).round(1)
        seg_counts['Label'] = (
            seg_counts['Count'].astype(str) +
            ' (' + seg_counts['Pct'].astype(str) + '%)'
        )
        fig = px.bar(
            seg_counts, x='Segment', y='Count',
            color='Segment',
            color_discrete_map=COLORS,
            text='Label'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-25,
            plot_bgcolor='white',
            height=420,
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Revenue Distribution by Segment")
        seg_rev = rfm_filtered.groupby('Segment').agg(
            Total_Revenue = ('Monetary', 'sum'),
            Avg_Revenue   = ('Monetary', 'mean'),
            Customers     = ('CustomerID', 'count')
        ).reset_index().round(2)

        fig2 = px.treemap(
            seg_rev,
            path=['Segment'],
            values='Total_Revenue',
            color='Avg_Revenue',
            color_continuous_scale='Teal',
            hover_data={
                'Customers': True,
                'Total_Revenue': ':,.0f',
                'Avg_Revenue': ':,.0f'
            }
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Insight Box ───────────────────────────────────
    champions_rev = rfm_filtered[
        rfm_filtered['Segment'] == 'Champions'
    ]['Monetary'].sum()
    total_rev = rfm_filtered['Monetary'].sum()
    champ_pct = champions_rev / total_rev * 100

    st.markdown(f"""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong>
        Champions represent only
        <strong>
        {(rfm_filtered['Segment']=='Champions').mean()*100:.1f}%
        </strong>
        of customers but generate
        <strong>£{champions_rev:,.0f}
        ({champ_pct:.1f}%)</strong>
        of total revenue — a textbook example of
        the Pareto Principle in e-commerce data.
    </div>
    """, unsafe_allow_html=True)

    # ── Charts Row 2 ──────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 RFM Score Distribution")
        fig3 = px.histogram(
            rfm_filtered, x='RFM_Total',
            nbins=13,
            color_discrete_sequence=['#667eea'],
            labels={'RFM_Total': 'RFM Total Score (3-15)'}
        )
        fig3.update_layout(
            plot_bgcolor='white', height=350,
            bargap=0.1,
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0',
                       title='Number of Customers')
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("🌍 Top 10 Countries by Revenue")
        top_countries = df_clean.groupby('Country')[
            'TotalPrice'
        ].sum().sort_values(ascending=False).head(10) \
         .reset_index()
        top_countries.columns = ['Country', 'Revenue']

        fig4 = px.bar(
            top_countries,
            x='Country', y='Revenue',
            color='Country',
            color_discrete_sequence=[
                '#E07A5F' if c == '"🇩🇪 Germany'
                else '#667eea'
                for c in top_countries['Country']
            ],
            text=top_countries['Revenue'].apply(
                lambda x: f'£{x/1000:.0f}K'
            )
        )
        fig4.update_traces(textposition='outside',
                           showlegend=False)
        fig4.update_layout(
            plot_bgcolor='white', height=350,
            xaxis_tickangle=-30,
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Full Data Table ───────────────────────────────
    st.markdown("---")
    st.subheader("📋 Customer Data Table")

    show_cols = ['CustomerID', 'Country', 'Recency',
                 'Frequency', 'Monetary',
                 'RFM_Score', 'Segment', 'CLV_Tier']
    st.dataframe(
        rfm_filtered[show_cols]
        .sort_values('Monetary', ascending=False),
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # ── Download ──────────────────────────────────────
    csv = rfm_filtered[show_cols].to_csv(index=False)
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name='rfm_segments_filtered.csv',
        mime='text/csv'
    )

# ══════════════════════════════════════════════════════
# PAGE 2: GERMAN DEEP DIVE
# ══════════════════════════════════════════════════════
elif page == "🏆 German Deep Dive":

    st.markdown("""
    <div class="main-header">
        <h1>German Market Deep Dive</h1>
        <p>107 German Customers · 3.6x Higher CLV ·
           Statistical Significance Proven (p&lt;0.05)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Germany KPIs ──────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("German Customers", f"{len(rfm_germany):,}")
    k2.metric("German Revenue",
              f"£{rfm_germany['Monetary'].sum():,.0f}")
    k3.metric("Avg Spend (DE)",
              f"£{rfm_germany['Monetary'].mean():,.0f}",
              f"+£{rfm_germany['Monetary'].mean() - rfm['Monetary'].mean():,.0f} vs global")
    k4.metric("DE Champions",
              f"{(rfm_germany['Segment']=='Champions').sum()}",
              f"{(rfm_germany['Segment']=='Champions').mean()*100:.1f}% (vs 22.1% global)")
    k5.metric("DE Median CLV",
              f"{rfm_germany['CLV'].median():.1f}",
              f"3.6x global median")

    st.markdown("---")

    # ── Comparison Chart ──────────────────────────────
    st.subheader("🔍 "🇩🇪 Germany vs Global — Segment Distribution")

    global_s = (rfm['Segment'].value_counts() /
                len(rfm) * 100).round(1).reset_index()
    global_s.columns = ['Segment', 'Percentage']
    global_s['Market'] = '🌍 Global'

    german_s = (rfm_germany['Segment'].value_counts() /
                len(rfm_germany) * 100).round(1).reset_index()
    german_s.columns = ['Segment', 'Percentage']
    german_s['Market'] = '"🇩🇪 Germany'

    combined = pd.concat([global_s, german_s])

    fig5 = px.bar(
        combined, x='Segment', y='Percentage',
        color='Market', barmode='group',
        color_discrete_map={
            '🌍 Global':   '#6C7086',
            '"🇩🇪 Germany': '#2EC4B6'
        },
        text='Percentage',
        labels={'Percentage': '% of Customers'}
    )
    fig5.update_traces(
        texttemplate='%{text}%',
        textposition='outside'
    )
    fig5.update_layout(
        plot_bgcolor='white', height=450,
        xaxis_tickangle=-25,
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(gridcolor='#f0f0f0'),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong>
        German customers have <strong>29.9% Champions</strong>
        vs 22.1% globally — and only <strong>4.7% Hibernating</strong>
        vs 10.9% globally. Germany's customer base is significantly
        more active and loyal than the global average.
    </div>
    """, unsafe_allow_html=True)

    # ── Stats comparison table ────────────────────────
    st.markdown("---")
    st.subheader("📐 Statistical Comparison")

    comp_df = pd.DataFrame({
        'Metric': ['Avg Recency (days)',
                   'Avg Frequency',
                   'Avg Monetary (£)',
                   'Median CLV',
                   'Champions %'],
        '🌍 Global': [
            f"{rfm['Recency'].mean():.1f}",
            f"{rfm['Frequency'].mean():.1f}",
            f"£{rfm['Monetary'].mean():,.0f}",
            f"{rfm['CLV'].median():.1f}",
            f"{(rfm['Segment']=='Champions').mean()*100:.1f}%"
        ],
        '"🇩🇪 Germany': [
            f"{rfm_germany['Recency'].mean():.1f}",
            f"{rfm_germany['Frequency'].mean():.1f}",
            f"£{rfm_germany['Monetary'].mean():,.0f}",
            f"{rfm_germany['CLV'].median():.1f}",
            f"{(rfm_germany['Segment']=='Champions').mean()*100:.1f}%"
        ],
        'Significant?': [
            '✅ YES (p=0.0005)',
            '❌ NO (p=0.117)',
            '✅ YES (p=0.0000)',
            '✅ YES (3.6x)',
            '✅ YES (+7.8%)'
        ]
    })
    st.dataframe(comp_df, use_container_width=True,
                 hide_index=True)

    # ── Top German Customers ──────────────────────────
    st.markdown("---")
    st.subheader("🏆 Top 15 German Customers by CLV")
    top_de = rfm_germany.nlargest(15, 'CLV')[[
        'CustomerID', 'Recency', 'Frequency',
        'Monetary', 'CLV', 'CLV_Tier', 'Segment'
    ]].round(2)
    st.dataframe(top_de, use_container_width=True,
                 hide_index=True)

    csv_de = rfm_germany.to_csv(index=False)
    st.download_button(
        label="⬇️ Download German Customer Data",
        data=csv_de,
        file_name='rfm_germany.csv',
        mime='text/csv'
    )

# ══════════════════════════════════════════════════════
# PAGE 3: CLV ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "💎 CLV Analysis":

    st.markdown("""
    <div class="main-header">
        <h1>💎 Customer Lifetime Value Analysis</h1>
        <p>CLV = (Frequency × Monetary) / Recency ·
           Identifies highest future-value customers</p>
    </div>
    """, unsafe_allow_html=True)

    # ── CLV KPIs ──────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg CLV (Global)",
              f"{rfm['CLV'].mean():,.0f}")
    k2.metric("Avg CLV ("🇩🇪 Germany)",
              f"{rfm_germany['CLV'].mean():,.0f}",
              "Higher than global")
    k3.metric("Platinum Customers",
              f"{(rfm['CLV_Tier']=='Platinum').sum():,}")
    k4.metric("Top Customer CLV",
              f"{rfm['CLV'].max():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💎 Global CLV Tier Distribution")
        clv_counts = rfm['CLV_Tier'] \
            .value_counts().reset_index()
        clv_counts.columns = ['Tier', 'Count']
        tier_order = ['Bronze', 'Silver',
                      'Gold', 'Platinum']
        clv_colors = {
            'Bronze': '#CD7F32', 'Silver': '#C0C0C0',
            'Gold': '#FFD700', 'Platinum': '#2EC4B6'
        }
        fig6 = px.bar(
            clv_counts,
            x='Tier', y='Count',
            color='Tier',
            color_discrete_map=clv_colors,
            text='Count',
            category_orders={'Tier': tier_order}
        )
        fig6.update_traces(textposition='outside')
        fig6.update_layout(
            showlegend=False,
            plot_bgcolor='white', height=380
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        st.subheader("German CLV Tier Distribution")
        de_clv = rfm_germany['CLV_Tier'] \
            .value_counts().reset_index()
        de_clv.columns = ['Tier', 'Count']
        fig7 = px.bar(
            de_clv,
            x='Tier', y='Count',
            color='Tier',
            color_discrete_map=clv_colors,
            text='Count',
            category_orders={'Tier': tier_order}
        )
        fig7.update_traces(textposition='outside')
        fig7.update_layout(
            showlegend=False,
            plot_bgcolor='white', height=380
        )
        st.plotly_chart(fig7, use_container_width=True)

    # ── Top 20 CLV Customers ──────────────────────────
    st.markdown("---")
    st.subheader("🏆 Top 20 Highest CLV Customers")
    top20 = rfm.nlargest(20, 'CLV')[[
        'CustomerID', 'Country', 'Recency',
        'Frequency', 'Monetary',
        'CLV', 'CLV_Tier', 'Segment'
    ]].round(2)
    st.dataframe(top20, use_container_width=True,
                 hide_index=True)

# ══════════════════════════════════════════════════════
# PAGE 4: COHORT RETENTION
# ══════════════════════════════════════════════════════
elif page == "📅 Cohort Retention":

    st.markdown("""
    <div class="main-header">
        <h1>📅 Cohort Retention Analysis</h1>
        <p>Tracks what % of customers return each month
           after their first purchase</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Build cohort data ─────────────────────────────
    with st.spinner('Building cohort matrix...'):
        df_c = df_clean.copy()
        df_c['InvoiceMonth'] = \
            df_c['InvoiceDate'].dt.to_period('M')
        first_purchase = df_c.groupby('CustomerID')[
            'InvoiceMonth'
        ].min().reset_index()
        first_purchase.columns = ['CustomerID',
                                   'CohortMonth']
        df_c = df_c.merge(first_purchase,
                          on='CustomerID')
        df_c['CohortIndex'] = (
            df_c['InvoiceMonth'] -
            df_c['CohortMonth']
        ).apply(lambda x: x.n)

        cohort_data = df_c.groupby(
            ['CohortMonth', 'CohortIndex']
        )['CustomerID'].nunique().reset_index()

        cohort_counts = cohort_data.pivot_table(
            index='CohortMonth',
            columns='CohortIndex',
            values='CustomerID'
        )
        cohort_size = cohort_counts.iloc[:, 0]
        retention = cohort_counts.divide(
            cohort_size, axis=0
        ) * 100

    # ── KPIs ──────────────────────────────────────────
    avg_m1 = retention.iloc[:, 1].mean()
    avg_m12 = retention[12].mean() \
        if 12 in retention.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cohorts", f"{len(retention)}")
    k2.metric("Avg Month-1 Retention",
              f"{avg_m1:.1f}%")
    k3.metric("Avg Month-12 Retention",
              f"{avg_m12:.1f}%",
              "Annual Christmas spike!")
    k4.metric("Longest Cohort",
              "24 months")

    st.markdown("---")

    # ── Heatmap ───────────────────────────────────────
    st.subheader("🗺️ Retention Heatmap")
    st.caption(
        "Each cell = % of customers from that cohort "
        "who returned in that month. "
        "Darker = higher retention."
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig8, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        retention.round(1),
        annot=True, fmt='.0f',
        cmap='YlOrRd',
        linewidths=0.3,
        linecolor='white',
        ax=ax,
        vmin=0, vmax=50,
        annot_kws={'size': 7},
        cbar_kws={'label': 'Retention Rate (%)'}
    )
    ax.set_title(
        'Customer Cohort Retention Matrix\n'
        'Month 0 = First Purchase (always 100%)',
        fontsize=12, fontweight='bold', pad=15
    )
    ax.set_xlabel(
        'Months Since First Purchase',
        fontsize=10
    )
    ax.set_ylabel(
        'Cohort (First Purchase Month)',
        fontsize=10
    )
    ax.set_yticklabels(
        [str(l) for l in retention.index],
        rotation=0, fontsize=7
    )
    plt.tight_layout()
    st.pyplot(fig8)

    # ── Retention Curve ───────────────────────────────
    st.markdown("---")
    st.subheader("📈 Average Retention Curve")

    avg_ret = retention.mean(axis=0)
    months_plot = [m for m in
                   [0,1,2,3,4,5,6,9,12,15,18,21,24]
                   if m in avg_ret.index]
    values_plot = [avg_ret[m] for m in months_plot]

    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(
        x=months_plot, y=values_plot,
        mode='lines+markers+text',
        line=dict(color='#2EC4B6', width=3),
        marker=dict(size=10, color='white',
                    line=dict(color='#2EC4B6',
                              width=2.5)),
        text=[f'{v:.0f}%' for v in values_plot],
        textposition='top center',
        fill='tozeroy',
        fillcolor='rgba(46,196,182,0.1)'
    ))
    fig9.add_hline(
        y=25, line_dash='dash',
        line_color='#F4A261',
        annotation_text='Industry avg ~25%'
    )
    fig9.update_layout(
        plot_bgcolor='white', height=380,
        xaxis=dict(title='Months Since First Purchase',
                   gridcolor='#f0f0f0'),
        yaxis=dict(title='Avg Retention Rate (%)',
                   gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Key Insight:</strong>
        After the initial sharp drop at Month 1,
        retention <strong>stabilises at 18-22%</strong>
        and even <strong>rises at Month 12</strong>
        — confirming an annual Christmas shopping pattern.
        Getting the second purchase is the hardest step.
    </div>
    """, unsafe_allow_html=True)
