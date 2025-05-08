import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dự báo chi phí khám chữa bệnh", layout="centered")
st.title("📈 Dự báo Chi phí Khám chữa bệnh theo nhóm")

# === 1. Upload file Excel ===
uploaded_file = st.file_uploader("📤 Tải lên file Excel (.xlsx)", type="xlsx")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name='ChiPhi')
    except:
        st.error("❌ Không tìm thấy sheet 'ChiPhi'. Vui lòng kiểm tra lại file.")
    else:
        # === 2. Chuẩn bị dữ liệu ===
        df['ThoiGian'] = df['Năm'].astype(str) + ' Q' + df['Quý'].astype(str).str.extract('(\d+)')[0]
        df['Index'] = range(1, len(df) + 1)

        st.success("✅ Đã nạp dữ liệu. Vui lòng chọn nhóm chi phí bên dưới.")
        nhoms = df['Nhóm chi phí'].unique()
        chon_nhom = st.selectbox("🔍 Chọn nhóm chi phí cần dự báo:", nhoms)

        df_nhom = df[df['Nhóm chi phí'] == chon_nhom].copy()

        # === 3. Huấn luyện mô hình Linear Regression ===
        X = df_nhom[['Index']]
        y_bhyt = df_nhom['BHYT_TT']
        y_chiphi = df_nhom['Chi phí bình quân']

        model_bhyt = LinearRegression().fit(X, y_bhyt)
        model_chiphi = LinearRegression().fit(X, y_chiphi)

        next_index = X.iloc[-1, 0] + 1
        du_doan_bhyt = model_bhyt.predict(pd.DataFrame({'Index': [next_index]}))[0]
        du_doan_chiphi = model_chiphi.predict(pd.DataFrame({'Index': [next_index]}))[0]

        # === 4. Hiển thị kết quả ===
        st.subheader(f"📊 Dự báo quý tiếp theo ({chon_nhom})")
        st.markdown(f"""
        - **➡️ BHYT_TT:** `{round(du_doan_bhyt):,}` VND  
        - **➡️ Chi phí bình quân:** `{round(du_doan_chiphi):,}` VND  
        """)

        # === 5. Vẽ biểu đồ xu hướng + hiển thị nhãn số ===
        x_labels = df_nhom['ThoiGian'].tolist() + ['Dự báo']
        y_bhyt_all = y_bhyt.tolist() + [du_doan_bhyt]
        y_chiphi_all = y_chiphi.tolist() + [du_doan_chiphi]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_labels, y_bhyt_all, marker='o', label='BHYT_TT')
        ax.plot(x_labels, y_chiphi_all, marker='x', label='Chi phí bình quân')

        # Thêm số lên điểm dự báo
        ax.annotate(f"{du_doan_bhyt/1e6:.1f}M",
                    xy=(len(x_labels)-1, du_doan_bhyt),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='red')

        ax.annotate(f"{du_doan_chiphi:,.0f}",
                    xy=(len(x_labels)-1, du_doan_chiphi),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='blue')

        ax.set_title(f'Dự báo chi phí theo thời gian - Nhóm: {chon_nhom}')
        ax.set_xlabel('Thời điểm (Quý/Năm)')
        ax.set_ylabel('Giá trị (Triệu VND)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M'))

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # === 6. Thêm dòng dữ liệu dự báo vào bảng ===
        du_bao_nam = df_nhom['Năm'].max()
        du_bao_quy_cu = int(df_nhom['Quý'].iloc[-1][-1])
        if du_bao_quy_cu == 4:
            du_bao_nam += 1
            du_bao_quy = 'Q1'
        else:
            du_bao_quy = f"Q{du_bao_quy_cu + 1}"

        df_du_bao = pd.DataFrame({
            'Năm': [du_bao_nam],
            'Quý': [du_bao_quy],
            'Nhóm chi phí': [chon_nhom],
            'BHYT_TT': [round(du_doan_bhyt)],
            'Chi phí bình quân': [round(du_doan_chiphi)],
            'ThoiGian': ['Dự báo'],
            'Index': [next_index]
        })

        df_hien_thi = pd.concat([df_nhom, df_du_bao], ignore_index=True)

        # === 7. Hiển thị bảng dữ liệu (có dòng dự báo) ===
        with st.expander("📂 Xem dữ liệu gốc + dự báo"):
            st.dataframe(df_hien_thi)

else:
    st.info("⬆️ Hãy tải lên file Excel có sheet tên 'ChiPhi'. File cần có các cột: Năm, Quý, Nhóm chi phí, BHYT_TT, Chi phí bình quân.")
