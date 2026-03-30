import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard PHTB - Kelompok 2", page_icon="🏠", layout="wide")

# ==========================
# Load Artifacts
# ==========================
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'rf_model.pkl'))
    lr_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'lr_model.pkl'))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    le_dict = joblib.load(os.path.join(ARTIFACTS_DIR, 'le_dict.pkl'))
    config = joblib.load(os.path.join(ARTIFACTS_DIR, 'config.pkl'))
    return rf_model, lr_model, scaler, le_dict, config


@st.cache_data
def load_dashboard_data():
    return pd.read_parquet(os.path.join(ARTIFACTS_DIR, 'dashboard_data.parquet'))


try:
    rf_model, lr_model, scaler, le_dict, config = load_artifacts()
    df = load_dashboard_data()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

if not model_loaded:
    st.error(
        "**Artifacts belum tersedia.** Jalankan cell 6.1 di notebook terlebih dahulu."
    )
    st.stop()

# ==========================
# Sidebar Filters
# ==========================
with st.sidebar:
    st.header("🔎 Filter Data")

    # KPP ADM PENJUAL
    kpp_list = sorted(df['KPPADM_PENJUAL'].dropna().unique().tolist())
    selected_kpp = st.multiselect(
        "KPP Adm Penjual",
        options=kpp_list,
        default=[],
        placeholder="Semua KPP"
    )

    # Tahun Transaksi
    tahun_list = sorted(df['TAHUN_TRANSAKSI'].unique().tolist())
    selected_tahun = st.multiselect(
        "Tahun Transaksi",
        options=tahun_list,
        default=[],
        placeholder="Semua Tahun"
    )

    # Jenis Objek Pajak
    jop_list = sorted(df['JENIS_OBJEK_PAJAK'].dropna().unique().tolist())
    selected_jop = st.multiselect(
        "Jenis Objek Pajak",
        options=jop_list,
        default=[],
        placeholder="Semua Jenis"
    )

    # Jenis Penjual
    jnsp_list = sorted(df['JNS_PENJUAL'].dropna().unique().tolist())
    selected_jnsp = st.multiselect(
        "Jenis Penjual",
        options=jnsp_list,
        default=[],
        placeholder="Semua Jenis"
    )

    st.divider()
    st.caption("Kelompok 2 — PHTB Analytics")

# ==========================
# Apply Filters
# ==========================
df_filtered = df.copy()
if selected_kpp:
    df_filtered = df_filtered[df_filtered['KPPADM_PENJUAL'].isin(selected_kpp)]
if selected_tahun:
    df_filtered = df_filtered[df_filtered['TAHUN_TRANSAKSI'].isin(selected_tahun)]
if selected_jop:
    df_filtered = df_filtered[df_filtered['JENIS_OBJEK_PAJAK'].isin(selected_jop)]
if selected_jnsp:
    df_filtered = df_filtered[df_filtered['JNS_PENJUAL'].isin(selected_jnsp)]

# ==========================
# Title
# ==========================
st.title("🏠 Dashboard Analisis Transaksi PHTB")
st.markdown("Data transaksi Pengalihan Hak atas Tanah dan/atau Bangunan (PHTB) tahun **2020–2024**")

if selected_kpp or selected_tahun or selected_jop or selected_jnsp:
    filters_text = []
    if selected_kpp:
        filters_text.append(f"KPP: {', '.join(selected_kpp[:3])}{'...' if len(selected_kpp) > 3 else ''}")
    if selected_tahun:
        filters_text.append(f"Tahun: {', '.join(map(str, selected_tahun))}")
    if selected_jop:
        filters_text.append(f"Objek: {', '.join(selected_jop[:2])}{'...' if len(selected_jop) > 2 else ''}")
    if selected_jnsp:
        filters_text.append(f"Penjual: {', '.join(selected_jnsp)}")
    st.info(f"🔎 Filter aktif: {' | '.join(filters_text)}")

# ==========================
# 1. KPI Cards - Total Keseluruhan
# ==========================
st.subheader("📊 Ringkasan Total")

total_trx = len(df_filtered)
total_harga = df_filtered['HARGA'].sum()
total_pph_terutang = df_filtered['JML_PPHFINAL_TERUTANG'].sum()
total_pph_bayar = df_filtered['JUMLAH_BAYAR'].sum()
selisih_pph = total_pph_terutang - total_pph_bayar

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Jumlah Transaksi", f"{total_trx:,.0f}")
col2.metric("Total Harga (Rp)", f"{total_harga / 1e12:,.2f} T")
col3.metric("PPh Terutang (Rp)", f"{total_pph_terutang / 1e12:,.2f} T")
col4.metric("PPh Dibayar (Rp)", f"{total_pph_bayar / 1e12:,.2f} T")
col5.metric("Selisih PPh (Rp)", f"{selisih_pph / 1e9:,.2f} M",
            delta=f"{selisih_pph / total_pph_terutang * 100:.1f}%" if total_pph_terutang > 0 else "0%",
            delta_color="inverse")

st.divider()

# ==========================
# 2. Tabel & Grafik Per Tahun Transaksi
# ==========================
st.subheader("📅 Ringkasan Per Tahun Transaksi")

df_tahun = df_filtered.groupby('TAHUN_TRANSAKSI').agg(
    Jumlah_Transaksi=('HARGA', 'count'),
    Total_Harga=('HARGA', 'sum'),
    PPh_Terutang=('JML_PPHFINAL_TERUTANG', 'sum'),
    PPh_Dibayar=('JUMLAH_BAYAR', 'sum'),
    Rata2_Harga=('HARGA', 'mean'),
).reset_index()
df_tahun['Selisih_PPh'] = df_tahun['PPh_Terutang'] - df_tahun['PPh_Dibayar']

col_tbl, col_chart = st.columns([2, 3])

with col_tbl:
    df_display = df_tahun.copy()
    df_display.columns = ['Tahun', 'Jml Trx', 'Total Harga', 'PPh Terutang', 'PPh Dibayar', 'Rata2 Harga', 'Selisih PPh']
    st.dataframe(
        df_display.style.format({
            'Jml Trx': '{:,.0f}',
            'Total Harga': 'Rp {:,.0f}',
            'PPh Terutang': 'Rp {:,.0f}',
            'PPh Dibayar': 'Rp {:,.0f}',
            'Rata2 Harga': 'Rp {:,.0f}',
            'Selisih PPh': 'Rp {:,.0f}',
        }),
        use_container_width=True,
        hide_index=True
    )

with col_chart:
    fig_tahun = go.Figure()
    fig_tahun.add_trace(go.Bar(
        x=df_tahun['TAHUN_TRANSAKSI'].astype(str),
        y=df_tahun['PPh_Terutang'],
        name='PPh Terutang',
        marker_color='#2E86C1'
    ))
    fig_tahun.add_trace(go.Bar(
        x=df_tahun['TAHUN_TRANSAKSI'].astype(str),
        y=df_tahun['PPh_Dibayar'],
        name='PPh Dibayar',
        marker_color='#27AE60'
    ))
    fig_tahun.add_trace(go.Scatter(
        x=df_tahun['TAHUN_TRANSAKSI'].astype(str),
        y=df_tahun['Jumlah_Transaksi'],
        name='Jumlah Transaksi',
        yaxis='y2',
        mode='lines+markers',
        marker_color='#E74C3C',
        line=dict(width=2)
    ))
    fig_tahun.update_layout(
        title='PPh Terutang vs Dibayar & Jumlah Transaksi per Tahun',
        yaxis=dict(title='Rupiah (Rp)'),
        yaxis2=dict(title='Jumlah Transaksi', overlaying='y', side='right'),
        barmode='group',
        legend=dict(orientation='h', y=-0.15),
        height=400
    )
    st.plotly_chart(fig_tahun, use_container_width=True)

st.divider()

# ==========================
# 3. Ringkasan Per KPP ADM PENJUAL
# ==========================
st.subheader("🏢 Ringkasan Per KPP Adm Penjual")

df_kpp = df_filtered.groupby('KPPADM_PENJUAL').agg(
    Jumlah_Transaksi=('HARGA', 'count'),
    Total_Harga=('HARGA', 'sum'),
    PPh_Terutang=('JML_PPHFINAL_TERUTANG', 'sum'),
    PPh_Dibayar=('JUMLAH_BAYAR', 'sum'),
).reset_index()
df_kpp['Selisih_PPh'] = df_kpp['PPh_Terutang'] - df_kpp['PPh_Dibayar']
df_kpp = df_kpp.sort_values('Total_Harga', ascending=False)

col_kpp_tbl, col_kpp_chart = st.columns([2, 3])

with col_kpp_tbl:
    df_kpp_display = df_kpp.copy()
    df_kpp_display.columns = ['KPP Adm Penjual', 'Jml Trx', 'Total Harga', 'PPh Terutang', 'PPh Dibayar', 'Selisih PPh']
    st.dataframe(
        df_kpp_display.style.format({
            'Jml Trx': '{:,.0f}',
            'Total Harga': 'Rp {:,.0f}',
            'PPh Terutang': 'Rp {:,.0f}',
            'PPh Dibayar': 'Rp {:,.0f}',
            'Selisih PPh': 'Rp {:,.0f}',
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )

with col_kpp_chart:
    top_n = 15
    df_kpp_top = df_kpp.head(top_n)
    fig_kpp = px.bar(
        df_kpp_top,
        x='Total_Harga',
        y='KPPADM_PENJUAL',
        orientation='h',
        title=f'Top {top_n} KPP Adm Penjual (Total Harga Transaksi)',
        labels={'Total_Harga': 'Total Harga (Rp)', 'KPPADM_PENJUAL': 'KPP'},
        color='Jumlah_Transaksi',
        color_continuous_scale='Blues',
    )
    fig_kpp.update_layout(yaxis=dict(autorange='reversed'), height=400)
    st.plotly_chart(fig_kpp, use_container_width=True)

st.divider()

# ==========================
# 4. Grafik Distribusi
# ==========================
st.subheader("📈 Grafik Distribusi")

tab1, tab2, tab3 = st.tabs(["Jenis Objek Pajak", "Jenis Penjual", "Distribusi Harga"])

with tab1:
    col_pie, col_bar = st.columns(2)
    df_jop = df_filtered.groupby('JENIS_OBJEK_PAJAK').agg(
        Jumlah=('HARGA', 'count'),
        Total_Harga=('HARGA', 'sum'),
        PPh_Terutang=('JML_PPHFINAL_TERUTANG', 'sum'),
    ).reset_index()

    with col_pie:
        fig_pie = px.pie(
            df_jop, values='Jumlah', names='JENIS_OBJEK_PAJAK',
            title='Komposisi Transaksi per Jenis Objek Pajak',
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_jop = px.bar(
            df_jop.sort_values('PPh_Terutang', ascending=True),
            x='PPh_Terutang', y='JENIS_OBJEK_PAJAK',
            orientation='h',
            title='PPh Terutang per Jenis Objek Pajak',
            labels={'PPh_Terutang': 'PPh Terutang (Rp)', 'JENIS_OBJEK_PAJAK': ''},
            color='PPh_Terutang', color_continuous_scale='Reds',
        )
        fig_jop.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_jop, use_container_width=True)

with tab2:
    col_p1, col_p2 = st.columns(2)
    df_jnsp = df_filtered.groupby('JNS_PENJUAL').agg(
        Jumlah=('HARGA', 'count'),
        Total_Harga=('HARGA', 'sum'),
        PPh_Terutang=('JML_PPHFINAL_TERUTANG', 'sum'),
    ).reset_index()

    with col_p1:
        fig_pie2 = px.pie(
            df_jnsp, values='Jumlah', names='JNS_PENJUAL',
            title='Komposisi Transaksi per Jenis Penjual',
            hole=0.4
        )
        fig_pie2.update_layout(height=400)
        st.plotly_chart(fig_pie2, use_container_width=True)

    with col_p2:
        # Stacked bar per tahun per jenis penjual
        df_jnsp_yr = df_filtered.groupby(['TAHUN_TRANSAKSI', 'JNS_PENJUAL']).agg(
            Jumlah=('HARGA', 'count')
        ).reset_index()
        fig_stack = px.bar(
            df_jnsp_yr,
            x='TAHUN_TRANSAKSI', y='Jumlah', color='JNS_PENJUAL',
            title='Jumlah Transaksi per Tahun & Jenis Penjual',
            barmode='stack',
            labels={'TAHUN_TRANSAKSI': 'Tahun', 'Jumlah': 'Jumlah Transaksi'}
        )
        fig_stack.update_layout(height=400)
        st.plotly_chart(fig_stack, use_container_width=True)

with tab3:
    # Histogram harga (cap at 99th percentile for better view)
    cap_val = df_filtered['HARGA'].quantile(0.99)
    df_hist = df_filtered[df_filtered['HARGA'].between(1, cap_val)]
    fig_hist = px.histogram(
        df_hist, x='HARGA', nbins=100,
        title='Distribusi Harga Transaksi (s.d. Persentil 99)',
        labels={'HARGA': 'Harga (Rp)', 'count': 'Frekuensi'},
        color_discrete_sequence=['#2E86C1']
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ==========================
# 5. Cek Kewajaran Harga Transaksi
# ==========================
st.subheader("🔮 Cek Kewajaran Harga Transaksi")
st.markdown("Masukkan data transaksi untuk memprediksi harga wajar menggunakan model ML.")

kpp_classes = sorted(le_dict['KPP_LOKASI_OBJEK'].classes_.tolist())
jp_classes = sorted(le_dict['JENIS_PENGALIHAN'].classes_.tolist())
jop_classes = sorted(le_dict['JENIS_OBJEK_PAJAK'].classes_.tolist())
jnsp_classes = sorted(le_dict['JNS_PENJUAL'].classes_.tolist())

col_in1, col_in2 = st.columns(2)

with col_in1:
    inp_luas_tanah = st.number_input("Luas Tanah (m²)", min_value=1, max_value=100000, value=200, key="pred_lt")
    inp_luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0, max_value=50000, value=100, key="pred_lb")
    inp_tarif = st.selectbox("Tarif PPh (%)", [1.0, 2.5, 4.0], index=1, key="pred_tarif")
    inp_tahun = st.selectbox("Tahun Transaksi", [2020, 2021, 2022, 2023, 2024], index=4, key="pred_tahun")

with col_in2:
    inp_jp = st.selectbox("Jenis Pengalihan", jp_classes, key="pred_jp")
    inp_jop = st.selectbox("Jenis Objek Pajak", jop_classes, key="pred_jop")
    inp_jnsp = st.selectbox("Jenis Penjual", jnsp_classes, key="pred_jnsp")
    inp_kpp = st.selectbox("KPP Lokasi Objek", kpp_classes, key="pred_kpp")

inp_harga_aktual = st.number_input(
    "💰 Harga Transaksi Aktual (Rp) — *opsional, untuk analisis kewajaran*",
    min_value=0, value=0, step=1_000_000, key="pred_harga"
)

if st.button("🔍 Cek Harga Wajar", type="primary", use_container_width=True):
    # Encode
    input_cat = pd.DataFrame({
        'JENIS_PENGALIHAN': [inp_jp],
        'JENIS_OBJEK_PAJAK': [inp_jop],
        'JNS_PENJUAL': [inp_jnsp],
        'KPP_LOKASI_OBJEK': [inp_kpp],
    })
    for col_name in config['features_cat']:
        input_cat[col_name] = le_dict[col_name].transform(input_cat[col_name])

    input_num = pd.DataFrame({
        'LUAS_TANAH': [inp_luas_tanah],
        'LUAS_BANGUNAN': [inp_luas_bangunan],
        'TARIF_PPH': [inp_tarif],
        'TAHUN_TRANSAKSI': [inp_tahun],
    })

    input_raw = pd.concat([input_cat, input_num], axis=1)
    input_scaled = input_raw.copy()
    input_scaled[config['features_num']] = scaler.transform(input_raw[config['features_num']])

    pred_rf = rf_model.predict(input_scaled)[0]
    pred_lr = lr_model.predict(input_scaled)[0]

    col_rf, col_lr = st.columns(2)
    with col_rf:
        st.metric(
            "Random Forest (Model Utama)",
            f"Rp {pred_rf:,.0f}",
            help=f"R²={config['rf_metrics']['r2']:.4f}"
        )
    with col_lr:
        st.metric(
            "Linear Regression",
            f"Rp {pred_lr:,.0f}",
            help=f"R²={config['lr_metrics']['r2']:.4f}"
        )

    if inp_harga_aktual > 0:
        selisih = inp_harga_aktual - pred_rf
        selisih_pct = (selisih / pred_rf) * 100 if pred_rf != 0 else 0
        threshold = config['threshold_pct'] * 100

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Harga Aktual", f"Rp {inp_harga_aktual:,.0f}")
        col_b.metric("Harga Prediksi (RF)", f"Rp {pred_rf:,.0f}")
        col_c.metric("Selisih", f"Rp {selisih:,.0f}", f"{selisih_pct:+.1f}%")

        if selisih_pct < threshold:
            st.error(
                f"⚠️ **ANOMALI TERDETEKSI** (threshold: {threshold:.1f}%) — "
                "Harga transaksi **jauh di bawah** prediksi. Potensi **under-reporting**."
            )
        elif selisih_pct < -30:
            st.warning("⚠️ **PERHATIAN** — Harga di bawah prediksi. Perlu analisis lanjutan.")
        else:
            st.success("✅ Harga transaksi dalam rentang **wajar** menurut model.")
