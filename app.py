
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dự báo chi phí khám chữa bệnh", layout="centered")
st.title("📈 Dự báo Chi phí Khám chữa bệnh theo nhóm")

uploaded_file = st.file_uploader("📤 Tải lên file Excel (.xlsx)", type="xlsx")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name='ChiPhi')
    except:
        st.error("❌ Không tìm thấy sheet 'ChiPhi'. Vui lòng kiểm tra lại file.")
    else:
        df['ThoiGian'] = df['Năm'].astype(str) + ' Q' + df['Quý'].astype(str).str.extract('(\d+)')[0]
        df['Index'] = range(1, len(df) + 1)

        st.success("✅ Đã nạp dữ liệu. Vui lòng chọn nhóm chi phí bên dưới.")
        nhoms = df['Nhóm chi phí'].unique()
        chon_nhom = st.selectbox("🔍 Chọn nhóm chi phí cần dự báo:", nhoms)

        df_nhom = df[df['Nhóm chi phí'] == chon_nhom].copy()
        X = df_nhom[['Index']]
        y_bhyt = df_nhom['BHYT_TT']
        y_chiphi = df_nhom['Chi phí bình quân']

        model_bhyt = LinearRegression().fit(X, y_bhyt)
        model_chiphi = LinearRegression().fit(X, y_chiphi)

        next_index = X.iloc[-1, 0] + 1
        indices = [next_index, next_index + 1, next_index + 2]
        du_bao_bhyt = model_bhyt.predict(pd.DataFrame({'Index': indices}))
        du_bao_chiphi = model_chiphi.predict(pd.DataFrame({'Index': indices}))

        quy_du_bao = ['Q2', 'Q3', 'Q4']
        nam_du_bao = df_nhom['Năm'].max()
        quy_hien_tai = int(df_nhom['Quý'].iloc[-1][-1])
        quy_du_bao_chinh_xac = [(quy_hien_tai + i) % 4 or 4 for i in range(1, 4)]
        nam_du_bao_dieu_chinh = [nam_du_bao + ((quy_hien_tai + i - 1) // 4) for i in range(1, 4)]
        quy_labels = [f"Q{q}" for q in quy_du_bao_chinh_xac]

        df_du_bao_multi = pd.DataFrame({
            'Năm': nam_du_bao_dieu_chinh,
            'Quý': quy_labels,
            'Nhóm chi phí': [chon_nhom] * 3,
            'BHYT_TT': [round(v) for v in du_bao_bhyt],
            'Chi phí bình quân': [round(v) for v in du_bao_chiphi],
            'ThoiGian': [f"Dự báo {q}" for q in quy_labels],
            'Index': indices
        })

        df_hien_thi = pd.concat([df_nhom, df_du_bao_multi], ignore_index=True)
        st.subheader(f"📊 Dự báo 3 quý tiếp theo ({chon_nhom})")
        for i in range(3):
            st.markdown(f"- **➡️ {df_du_bao_multi['ThoiGian'][i]}**: BHYT_TT = `{df_du_bao_multi['BHYT_TT'][i]:,}` VND, Chi phí bình quân = `{df_du_bao_multi['Chi phí bình quân'][i]:,}` VND")

        x_labels = df_hien_thi['ThoiGian'].tolist()
        y_bhyt_all = df_hien_thi['BHYT_TT'].tolist()
        y_chiphi_all = df_hien_thi['Chi phí bình quân'].tolist()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_labels, y_bhyt_all, marker='o', label='BHYT_TT')
        ax.plot(x_labels, y_chiphi_all, marker='x', label='Chi phí bình quân')

        for i in range(3):
            ax.annotate(f"{du_bao_bhyt[i]/1e6:.1f}M", xy=(len(x_labels)-3+i, du_bao_bhyt[i]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, color='red')
            ax.annotate(f"{du_bao_chiphi[i]:,.0f}", xy=(len(x_labels)-3+i, du_bao_chiphi[i]), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9, color='blue')

        ax.set_title(f'Dự báo chi phí theo thời gian - Nhóm: {chon_nhom}')
        ax.set_xlabel('Thời điểm (Quý/Năm)')
        ax.set_ylabel('Giá trị (Triệu VND)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M'))
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        with st.expander("📂 Xem dữ liệu gốc + dự báo"):
            st.dataframe(df_hien_thi)
else:
    st.info("⬆️ Hãy tải lên file Excel có sheet tên 'ChiPhi'. File cần có các cột: Năm, Quý, Nhóm chi phí, BHYT_TT, Chi phí bình quân.")
