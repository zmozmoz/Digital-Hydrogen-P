import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RegularGridInterpolator
import io
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
import seaborn as sns
import joblib
import plotly.express as px


# 设置页面标题和图标
st.set_page_config(
    page_title="Digital Hydrogen-P",
    page_icon=r"D:\pythonProject3\streamlit\f8523a5d627f3875452fa1ece3b4d30.png",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown(
    """
    <style>
        div.stButton > button {
            width: 100%; 
            height: 40px; 
            font-size: 25px;
            background-color: #f0f2f6;  
            font-weight: bold;  
            border-radius: 20px;  
            border: none;
        }
        div.stButton > button:hover {
            background-color: #e3e5e7;
            opacity: 0.85;
        }   
        div.stButton > button:focus {
            background-color: #e3e5e7;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化 session_state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "show_sub_buttons" not in st.session_state:
    st.session_state.show_sub_buttons = False

if "df_results" not in st.session_state:
    st.session_state.df_results = None

if "show_plot" not in st.session_state:
    st.session_state.show_plot = False
# 在初始化 session_state 部分添加以下内容
if "units" not in st.session_state:
    st.session_state.units = {
        "pressure": "MPa",
        "temperature": "K",
        "thermal": "W/m·K",
        "viscosity": "μPa·s",
        "diffusion": "m²/s"
    }
if "show_experiment_sub_buttons" not in st.session_state:
    st.session_state.show_experiment_sub_buttons = False  # 记录“实验数据查询”是否展开

if "experiment_mode" not in st.session_state:
    st.session_state.experiment_mode = None

# 在数据展示前添加单位转换函数
def convert_units(value, prop_type):
    """根据全局设置进行单位转换"""
    units = st.session_state.units
    try:
        # 添加空值检查
        if value is None:
            raise ValueError("输入值不能为空")

        # 压力转换
        if prop_type == "pressure":
            if units["pressure"] == "bar":
                return value * 10, "bar"
            elif units["pressure"] == "kPa":
                return value * 1000, "kPa"
            else:
                return value, "MPa"

        # 温度转换（增加输入值验证）
        elif prop_type == "temperature":
            # 检查是否为合理温度值
            if units["temperature"] == "K" and value < 0:
                raise ValueError("开尔文温度不能小于0")

            if units["temperature"] == "°C":
                return value - 273.15, "°C"
            else:
                return value, "K"

        # 热导率转换
        elif prop_type == "thermal":
            if units["thermal"] == "mW/m·K":
                return value * 1000, "mW/m·K"
            else:
                return value, "W/m·K"

        # 粘度转换
        elif prop_type == "viscosity":
            if units["viscosity"] == "mPa·s":
                return value / 1000, "mPa·s"
            elif units["viscosity"] == "Pa·s":
                return value / 1e6, "Pa·s"
            else:
                return value, "μPa·s"

        # 扩散系数转换
        elif prop_type == "diffusion":
            if units["diffusion"] == "cm²/s":
                return value * 10000, "cm²/s"
            else:
                return value, "m²/s"

    except Exception as e:
        st.error(f"单位转换错误: {str(e)}")
        return value, "[ERROR]"


# 侧边栏处理
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_path = r"D:\pythonProject3\streamlit\8c3351f1e7b958ef4fdc8dfb9d5d99f.png"
your_base64_logo = image_to_base64(logo_path)

st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{your_base64_logo}" width="550"/>
        <h1 style="font-size: 27px; font-weight: bold; margin-top: 10px;">Digital Hydrogen-P</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# 侧边栏导航
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "🏠 Home"
    st.session_state.show_sub_buttons = False

if st.sidebar.button("⚙️ 功能"):
    st.session_state.page = "⚙️ 功能"
    st.session_state.show_sub_buttons = not st.session_state.show_sub_buttons

if st.session_state.show_sub_buttons:
    if st.sidebar.button("📌 定值查询"):
        st.session_state.page = "📌 定值查询"
    if st.sidebar.button("📏 范围查询"):
        st.session_state.page = "📏 范围查询"
    if st.sidebar.button("🔬 实验数据查询"):
        st.session_state.page = "🔬 实验数据查询"
        st.session_state.show_experiment_sub_buttons = not st.session_state.show_experiment_sub_buttons

    if st.session_state.page == "🔬 实验数据查询":
        with st.sidebar.expander("📂 选择子功能", expanded=st.session_state.show_experiment_sub_buttons):
            if st.sidebar.button("📊 查看实验数据"):
                st.session_state.experiment_mode = "实验数据"

            if st.sidebar.button("📈 实验数据的机器学习预测"):
                st.session_state.experiment_mode = "机器学习"


# 数据加载函数
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 清理所有列名空格
    return df

METHOD_MAPPING = {
    "griddata（线性插值）": "griddata",
    "RegularGridInterpolator（非线性插值）": "RegularGridInterpolator",
    "nearest（最近邻插值）": "nearest"
}
# 插值函数
def interpolate_property(pressure, temperature, data_df, property_name, method='griddata'):
    # 确保必要列存在
    required_cols = ['pressure', 'temperature', property_name]
    if not all(col in data_df.columns for col in required_cols):
        return "DataFrame missing required columns."

    for col in required_cols:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # 移除包含 NaN 的行
    #data_df.dropna(subset=required_cols, inplace=True)

    # 获取网格数据
    points = data_df[['pressure', 'temperature']].values
    values = data_df[property_name].values

    try:
        if method == 'griddata':  # 替代 interp2d
            result = griddata(points, values, (pressure, temperature), method='linear')
            return result if result is not None else "无法插值"
        elif method == 'RegularGridInterpolator':  # 原有方法
            pressure_vals = np.sort(data_df['pressure'].unique())
            temperature_vals = np.sort(data_df['temperature'].unique())
            property_matrix = data_df.pivot_table(index='temperature', columns='pressure', values=property_name).values
            interp_func = RegularGridInterpolator(
                (temperature_vals, pressure_vals),
                property_matrix,
                method='linear',
                bounds_error=False
            )
            return interp_func([[temperature, pressure]])[0]
        elif method == 'nearest':  # 另一种新的插值方式
            result = griddata(points, values, (pressure, temperature), method='nearest')
            return result if result is not None else "无法插值"
        else:
            return "Unsupported interpolation method."
    except Exception as e:
        return f"Interpolation failed with error: {str(e)}"


# 范围查询函数
def generate_table(min_pressure, max_pressure, step_size, min_temperature, max_temperature, method='interp2d'):
    # 将步长转换为浮点数以确保正确处理
    step_size = float(step_size)

    # 处理浮点数情况
    # 生成压力范围
    num_p = int(round((max_pressure - min_pressure) / step_size)) + 1
    pressures = []
    for p in np.linspace(min_pressure, max_pressure, num_p):
        if float(p).is_integer():
            pressures.append(int(p))
        else:
            pressures.append(round(p, 2))

    # 生成温度范围
    num_t = int(round((max_temperature - min_temperature) / step_size)) + 1
    temperatures = []
    for t in np.linspace(min_temperature, max_temperature, num_t):
        if float(t).is_integer():
            temperatures.append(int(t))
        else:
            temperatures.append(round(t, 2))

    thermal_df = load_data(r'D:\pythonProject3\streamlit\thermal_conductivity.csv')
    viscosity_df = load_data(r'D:\pythonProject3\streamlit\viscosity.csv')
    diffusion_df = load_data(r'D:\pythonProject3\streamlit\kuosanxishu.csv')

    table_data = []
    for pressure in pressures:
        for temperature in temperatures:
            row = {'Pressure': pressure, 'Temperature': temperature}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                # 热导率
                thermal_val = interpolate_property(pressure, temperature, thermal_df, prop, method)
                if isinstance(thermal_val, str):
                    st.error(f"热导率插值错误({pressure}MPa, {temperature}K, {prop}): {thermal_val}")
                    continue
                row[f'Thermal Conductivity {prop}'] = round(float(thermal_val), 3)

                # 粘度
                viscosity_val = interpolate_property(pressure, temperature, viscosity_df, prop, method)
                if isinstance(viscosity_val, str):
                    st.error(f"粘度插值错误({pressure}MPa, {temperature}K, {prop}): {viscosity_val}")
                    continue
                row[f'Viscosity {prop}'] = round(float(viscosity_val), 3)

                # 扩散系数
                diffusion_val = interpolate_property(pressure, temperature, diffusion_df, prop, method)
                if isinstance(diffusion_val, str):
                    st.error(f"扩散系数插值错误({pressure}MPa, {temperature}K, {prop}): {diffusion_val}")
                    continue
                row[f'Diffusion {prop}'] = round(float(diffusion_val), 9)

            table_data.append(row)
    return pd.DataFrame(table_data)


# 页面逻辑
if st.session_state.page == "🏠 Home":
    st.markdown("""
        <style>
            .block-container {
                padding-top: 20px; /* 适当减少上边距 */
            }
        </style>
    """, unsafe_allow_html=True)
    st.image(r"D:\pythonProject3\streamlit\4d8e4b158c47a8a9498496aa8bb051d.png", width=300)
    st.title("Digital Hydrogen-P")
    st.write("""
        **欢迎来到 Digital Hydrogen-P**  
        
        本网站致力于提供便捷、准确的氢气热物性数据查询服务。您可以快速进行数据查询、插值计算和可视化分析，深入了解氢气的热导率、粘度及扩散系数热物性信息。适用于科研人员、工程师及学生，帮助您更高效地完成氢气热物性相关的研究与工程设计。
    """)
    st.write("""
        **功能介绍**：
        - **定值查询**：轻松快速地输入指定压力和温度，精准获取氢气的热导率、粘度及扩散系数数值  
        - **范围查询**：支持自定义压力和温度范围及步长，批量获取氢气的热物性数据，并通过表格与图形直观展示结果，满足多种分析需求。  
        - **实验查询**：提供权威的实验数据来源，允许用户按文献标题选择和浏览热导率、粘度、扩散系数的实验数据，并支持便捷的数据导出功能。  
        - **图表展示**：内置交互式数据可视化工具，支持数据二维、三维可视化展示，帮助用户直观理解数据分布与趋势。  

        **机器学习势适用范围说明**  
        - **热导率** 　常规工况（≤120 MPa）：误差 ≤5% ✅ 　高压高温（>120 MPa, >440 K）：误差 ≤10% ✅️  
          高压低温（>120 MPa, ≤440 K）：最大误差 34% ⚠️  
        - **粘度** 　　宽压范围（≤180 MPa）：误差 ≤10% ✅ 　极端高压（200 MPa）：误差 ≤15% ⚠️  
        - **扩散系数** 　全工况：误差 ≤25% ✅  
    """)

    # st.write("""
    #     **示例文件下载**：
    #     [GitHub 参考资料](https://github.com/withand123/HydrogenCell-Life)
    # """)

elif st.session_state.page == "📏 范围查询":
    st.title("📏 范围查询")
    col1, col2 = st.columns(2)
    with col1:
        min_pressure = st.number_input("最小压力 (MPa) ", min_value=40.0, max_value=200.0, step=5.0, format="%.1f", value=None, help="请输入 40-200 MPa 之间的数值")
        max_pressure = st.number_input("最大压力 (MPa) ", min_value=min_pressure if min_pressure else 40.0, max_value=200.0, step=5.0,format="%.1f", value=None,help="请输入 40-200 MPa 之间的数值" )
    with col2:
        min_temperature = st.number_input("最小温度 (K) ", min_value=300.0, max_value=700.0, step=5.0, format="%.1f", value=None,help="请输入 300 - 700 K 之间的数值")
        max_temperature = st.number_input("最大温度 (K) ", min_value=min_temperature if min_temperature else 300.0, max_value=700.0,step=5.0, format="%.1f", value=None, help="请输入 300 - 700 K 之间的数值")

    step_size = st.selectbox("步长", [1, 2, 5, 10, 20, 50, 100])
    interpolation_method = st.selectbox(
        "插值方法",
        ["griddata（线性插值）", "RegularGridInterpolator（非线性插值）", "nearest（最近邻插值）"]
    )


    # 触发查询功能
    if st.button("🔍 查询"):
        actual_method = METHOD_MAPPING[interpolation_method]  # <-- 添加映射
        if min_pressure is None or max_pressure is None or min_temperature is None or max_temperature is None:
            st.warning("请输入完整的压力和温度范围")
        elif min_pressure > max_pressure or min_temperature > max_temperature:
            st.warning("最小压力不能大于最大压力，最小温度不能大于最大温度")
        else:
            st.session_state.df_results = generate_table(
                min_pressure, max_pressure, step_size, min_temperature, max_temperature, method=actual_method)
            if not st.session_state.df_results.empty:
                st.success("查询成功！")
            else:
                st.error("未生成有效查询结果，请检查输入参数或数据文件。")

    # 确保有查询结果后再显示功能
    if st.session_state.df_results is not None and not st.session_state.df_results.empty:
        # 创建格式化副本
        formatted_df = st.session_state.df_results.copy()
        # 压力列转换
        formatted_df['Pressure'] = formatted_df['Pressure'].apply(
            lambda x: convert_units(x, "pressure")[0]
        )

        # 温度列转换
        formatted_df['Temperature'] = formatted_df['Temperature'].apply(
            lambda x: convert_units(x, "temperature")[0]
        )

        # 其他列转换
        for col in formatted_df.columns:
            if 'Thermal Conductivity' in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: convert_units(x, "thermal")[0]
                )
            elif 'Viscosity' in col:
                formatted_df[col] = formatted_df[col].apply(lambda x: convert_units(x, "viscosity")[0])
            elif 'Diffusion' in col:
                formatted_df[col] = formatted_df[col].apply(lambda x: convert_units(x, "diffusion")[0])

        # 显示带单位标签的表格
        st.dataframe(formatted_df.style.format({
            'Pressure': '{:.2f} ' + st.session_state.units["pressure"],
            'Temperature': '{:.2f} ' + st.session_state.units["temperature"],**{col: '{:.4f} ' + st.session_state.units["thermal"]
               for col in formatted_df.columns if 'Thermal Conductivity' in col},**{col: '{:.4f} ' + st.session_state.units["viscosity"]
               for col in formatted_df.columns if 'Viscosity' in col},**{col: '{:.3e} ' + st.session_state.units["diffusion"]
               for col in formatted_df.columns if 'Diffusion' in col}}))

        # # 识别需要格式化的列
        # diffusion_cols = [col for col in formatted_df.columns if 'Diffusion' in col]
        #
        # # 应用科学计数法格式化
        # for col in diffusion_cols:
        #     formatted_df[col] = formatted_df[col].apply(
        #         lambda x: "{:.3e}".format(x) if isinstance(x, (int, float)) else x)
        #
        # # 显示带格式的表格
        # st.dataframe(formatted_df.style.set_properties(
        #     subset=diffusion_cols, ** {'text-align': 'center', 'font-family': 'monospace'}
        # ))
        def format_with_units_and_scientific(styler):
            units = st.session_state.units
            column_units = {}
            # 确定每个列的单位
            for col in styler.columns:
                if col == 'Pressure':
                    column_units[col] = units["pressure"]
                elif col == 'Temperature':
                    column_units[col] = units["temperature"]
                elif col.startswith('Thermal Conductivity'):
                    column_units[col] = units["thermal"]
                elif col.startswith('Viscosity'):
                    column_units[col] = units["viscosity"]
                elif col.startswith('Diffusion'):
                    column_units[col] = units["diffusion"]
                else:
                    # 跳过未知列
                    continue

            # 应用格式化
            for col, unit in column_units.items():
                for col, unit in column_units.items():
                    styler.format({col: lambda x, u=unit: (
                        f"{x:.3e} {u}" if (isinstance(x, (int, float)) and not pd.isna(x) and (abs(x) >= 1000 or (0 < abs(x) < 0.001)))
                        else (f"{x:.4f} {u}" if isinstance(x, (int, float)) and not pd.isna(x) else ""))}, na_rep="")
            return styler

        # 显示带单位和动态格式的表格
        #st.dataframe(formatted_df.style.pipe(format_with_units_and_scientific))
        # 按钮在同一行
        col1, col2, col3 = st.columns(3)

        with col1:
            # 生成 Excel 文件
            excel_data = io.BytesIO()
            with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
                st.session_state.df_results.to_excel(writer, index=False)
            excel_data.seek(0)  # 重要：重置指针位置
            st.download_button(
                label="📥 下载 Excel",
                data=excel_data,
                file_name="定值查询结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            # 生成 TXT 文件
            txt_data = st.session_state.df_results.to_csv(sep='\t', index=False).encode('utf-8')
            st.download_button(
                label="📥 下载 TXT",
                data=txt_data,
                file_name="定值查询结果.txt",
                mime="text/plain"
            )

        with col3:
            if st.button("📊 绘图"):
                st.session_state.show_plot = True

        # 结果可视化
        if st.session_state.show_plot:
            st.subheader("📊 结果可视化")

            plt.rcParams.update({'font.size': 16,  # 全局字体大小
                'axes.titlesize': 16,  # 子图标题大小
                'axes.labelsize': 15,  # 坐标轴标签大小
                'xtick.labelsize': 16,  # X轴刻度
                'ytick.labelsize': 16,  # Y轴刻度
                'legend.fontsize': 14  # 图例
            })

            # 让用户调整点的大小
            marker_size = st.slider("选择点的大小", min_value=2, max_value=10, value=5)

            # **增加图的大小**
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))  # 改为3行1列

            if min_temperature == max_temperature:
                x_axis = 'Pressure'
                x_label = "Pressure (MPa)"
            elif min_pressure == max_pressure:
                x_axis = 'Temperature'
                x_label = "Temperature (K)"
            else:
                x_axis = None

            if x_axis:
                for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                    axes[0].plot(
                        st.session_state.df_results[x_axis],
                        st.session_state.df_results[f'Thermal Conductivity {prop}'],
                        marker='o', markersize=marker_size, linestyle='-', label=f'Thermal {prop}'
                    )
                for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                    axes[1].plot(
                        st.session_state.df_results[x_axis],
                        st.session_state.df_results[f'Viscosity {prop}'],
                        marker='s', markersize=marker_size, linestyle='--', label=f'Viscosity {prop}'
                    )
                for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                    axes[2].plot(
                        st.session_state.df_results[x_axis],
                        st.session_state.df_results[f'Diffusion {prop}'],
                        marker='^', markersize=marker_size, linestyle=':', label=f'Diffusion {prop}'
                    )

                axes[0].set_xlabel(x_label)
                axes[0].set_ylabel("Thermal Conductivity (W/m·K)")
                axes[0].legend(loc='upper left')  # **图例固定在左上角**
                axes[0].set_title("热导率")

                axes[1].set_xlabel(x_label)
                axes[1].set_ylabel("Viscosity (μPa·s)")
                axes[1].legend(loc='upper left')  # **图例固定在左上角**
                axes[1].set_title("粘度")

                axes[2].set_xlabel(x_label)
                axes[2].set_ylabel("扩散系数 (m$^{2}$/s)")
                axes[2].legend(loc='upper left')  # **图例固定在左上角**
                axes[2].set_title("扩散系数")
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
                plt.tight_layout()
                st.pyplot(fig)


elif st.session_state.page == "📌 定值查询":
    st.title("📌 定值查询")

    # 新布局结构
    st.subheader("🔢 输入参数")

    # 第一行：压力温度输入框并排
    col_pres_temp = st.columns(2)

    with col_pres_temp[0]:
        pressure = st.number_input("输入压力 (MPa) ",min_value=40.0,max_value=200.0,step=5.0,format="%.2f",value=40.0,help="请输入 40 - 200 MPa 之间的数值")
    with col_pres_temp[1]:
        temperature = st.number_input("输入温度 (K) ",min_value=300.0, step=5.0,format="%.1f",value=300.0,help="请输入 300 - 700 K 之间的数值")

    # 第二行：插值方法和查询按钮并排
    col_method_btn = st.columns([2, 2])
    with col_method_btn[0]:
        interpolation_method = st.selectbox(
            "选择插值方法",
            ["griddata（线性插值）", "RegularGridInterpolator（非线性插值）", "nearest（最近邻插值）"],  # 带中文说明
            key="method_selectbox"
        )
    with col_method_btn[1]:
        st.write("")  # 垂直对齐
        st.write("")
        query_clicked = st.button("🔍 立即查询", use_container_width=True)

    # 触发查询按钮
    if query_clicked:
        if pressure is None or temperature is None:
            st.warning("请输入压力和温度")

        else:
            thermal_df = load_data(r'D:\pythonProject3\streamlit\thermal_conductivity.csv')
            viscosity_df = load_data(r'D:\pythonProject3\streamlit\viscosity.csv')
            diffusion_df = load_data(r'D:\pythonProject3\streamlit\kuosanxishu.csv')
            error_occurred = False

            actual_method = METHOD_MAPPING[interpolation_method]  # <-- 添加映射
            thermal_results = {}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                result = interpolate_property(pressure, temperature, thermal_df, prop, actual_method)
                if isinstance(result, str):
                    st.error(f"热导率 {prop} 计算错误: {result}")
                    error_occurred = True
                thermal_results[prop] = result

            viscosity_results = {}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                result = interpolate_property(pressure, temperature, viscosity_df, prop, actual_method)
                if isinstance(result, str):
                    st.error(f"粘度 {prop} 计算错误: {result}")
                    error_occurred = True
                viscosity_results[prop] = result

            diffusion_results = {}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                result = interpolate_property(pressure, temperature, diffusion_df, prop, actual_method)
                if isinstance(result, str):
                    st.error(f"扩散系数 {prop} 计算错误: {result}")
                    error_occurred = True
                diffusion_results[prop] = result

            if not error_occurred:
                st.session_state.thermal_results = thermal_results
                st.session_state.viscosity_results = viscosity_results
                st.session_state.diffusion_results = diffusion_results  # 存储扩散结果
                st.session_state.show_results = True

    if st.session_state.get("show_results", False):
        if st.session_state.get("show_results", False):
            # 单位转换
            pressure_display, pressure_unit = convert_units(pressure, "pressure")
            temp_display, temp_unit = convert_units(temperature, "temperature")

            st.subheader(f"当前参数：{pressure_display:.2f} {pressure_unit} | {temp_display:.2f} {temp_unit}")

            # 三列布局展示结果
            col_thermal, col_visc, col_diff = st.columns(3)

            with col_thermal:
                st.markdown(f"<h4 style='font-size:16px;'>热导率 ({st.session_state.units['thermal']})</h4>",
                            unsafe_allow_html=True)
                for name, val in st.session_state.thermal_results.items():
                    converted_val, _ = convert_units(val, "thermal")
                    st.write(f"**{name}**: {converted_val:.4f}")

            with col_visc:
                st.markdown(f"<h4 style='font-size:16px;'>粘度 ({st.session_state.units['viscosity']})</h4>",
                            unsafe_allow_html=True)
                for name, val in st.session_state.viscosity_results.items():
                    converted_val, _ = convert_units(val, "viscosity")
                    st.write(f"**{name}**: {converted_val:.4f}")

            with col_diff:
                st.markdown(f"<h4 style='font-size:16px;'>扩散系数 ({st.session_state.units['diffusion']})</h4>",
                            unsafe_allow_html=True)
                for name, val in st.session_state.diffusion_results.items():
                    converted_val, _ = convert_units(val, "diffusion")
                    st.write(f"**{name}**: {converted_val:.3e}")

elif st.session_state.page == "⚙️ 功能":
    st.title("⚙️ 全局单位设置")
    # 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("压力单位")
        st.session_state.units["pressure"] = st.selectbox(
            "选择压力单位",
            ["MPa", "bar", "kPa"],
            index=["MPa", "bar", "kPa"].index(st.session_state.units["pressure"])
        )

        st.subheader("温度单位")
        st.session_state.units["temperature"] = st.selectbox(
            "选择温度单位",
            ["K", "°C"],
            index=["K", "°C"].index(st.session_state.units["temperature"])
        )

    with col2:
        st.subheader("热导率单位")
        st.session_state.units["thermal"] = st.selectbox(
            "选择热导率单位",
            ["W/m·K", "mW/m·K"],
            index=["W/m·K", "mW/m·K"].index(st.session_state.units["thermal"])
        )

        st.subheader("粘度单位")
        st.session_state.units["viscosity"] = st.selectbox(
            "选择粘度单位",
            ["μPa·s", "mPa·s", "Pa·s"],
            index=["μPa·s", "mPa·s", "Pa·s"].index(st.session_state.units["viscosity"])
        )

        st.subheader("扩散系数单位")
        st.session_state.units["diffusion"] = st.selectbox(
            "选择扩散系数单位",
            ["m²/s", "cm²/s"],
            index=["m²/s", "cm²/s"].index(st.session_state.units["diffusion"])
        )

    st.success("单位设置已保存，所有查询结果将自动转换！")

elif st.session_state.page == "🔬 实验数据查询":
    st.title("🔬 实验数据查询")

    if st.session_state.experiment_mode == "实验数据":

        # 加载实验数据
        thermal_df = pd.read_csv(r'D:\pythonProject3\streamlit\shiyanredaol.csv')
        viscosity_df = pd.read_csv(r'D:\pythonProject3\streamlit\shiyanniandu.csv')

        # 获取所有文章标题
        thermal_article_titles = thermal_df['redaoarticle title'].unique()
        viscosity_article_titles = viscosity_df['nianduarticle title'].unique()

        # 选择热导率文章
        selected_thermal_article = st.selectbox("选择热导率文章", [""] + list(thermal_article_titles))

        # 选择粘度文章
        selected_viscosity_article = st.selectbox("选择粘度文章", [""] + list(viscosity_article_titles))

        if st.button("📊 绘制全部热导率实验数据的 3D 图"):
            fig = go.Figure(data=[go.Scatter3d(x=thermal_df['pressure'],y=thermal_df['temperature'],z=thermal_df['redaoexperimentalvalue'],mode='markers',marker=dict(size=5, color=thermal_df['redaoexperimentalvalue'], colorscale='Viridis')
            )])

            fig.update_layout(
                title="热导率实验数据 3D 可视化",scene=dict(xaxis_title="压力 (MPa)",yaxis_title="温度 (K)",zaxis_title="热导率实验值（W/(m·K)）")
            )

            st.plotly_chart(fig)

        if st.button("📊 绘制全部粘度实验数据的 3D 图"):
            fig = go.Figure(data=[go.Scatter3d(x=viscosity_df['pressure'],y=viscosity_df['temperature'],z=viscosity_df['nianduexperimentalvalue'],mode='markers',
                marker=dict(size=5, color=viscosity_df['nianduexperimentalvalue'], colorscale='Viridis')
            )])

            fig.update_layout(
                title="粘度实验数据 3D 可视化",
                scene=dict(xaxis_title="压力 (MPa)", yaxis_title="温度 (K)",zaxis_title="粘度实验值（μPa·s）")
            )

            st.plotly_chart(fig)

        table_data = None
        if selected_thermal_article:
            table_data = thermal_df[thermal_df['redaoarticle title'] == selected_thermal_article][
                ['pressure', 'temperature', 'redaoexperimentalvalue']]
            table_data.columns = ['Pressure', 'Temperature', 'Experimental Value']

        if selected_viscosity_article:
            table_data = viscosity_df[viscosity_df['nianduarticle title'] == selected_viscosity_article][
                ['pressure', 'temperature', 'nianduexperimentalvalue']]
            table_data.columns = ['Pressure', 'Temperature', 'Experimental Value']

        if table_data is not None and not table_data.empty:
            # 使用 st.data_editor() 代替 st.dataframe()
            st.data_editor(
                table_data.style.set_properties(**{'text-align': 'center'}),  # 设置文本居中
                height=400,  # 限制高度，启用滚动use_container_width=True  # 让表格自适应宽度
            )

            # **📥 下载功能**
            col1, col2 = st.columns(2)

            with col1:
                excel_data = io.BytesIO()
                with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
                    table_data.to_excel(writer, index=False)
                excel_data.seek(0)
                st.download_button("📥 下载 Excel", data=excel_data, file_name="实验数据.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with col2:
                txt_data = table_data.to_csv(sep='\t', index=False).encode('utf-8')
                st.download_button("📥 下载 TXT", data=txt_data, file_name="实验数据.txt", mime="text/plain")

            # **📊 绘制三维图**
            if st.button("📊 绘制 3D 图"):
                fig = go.Figure(data=[go.Scatter3d(x=table_data['Pressure'],y=table_data['Temperature'],z=table_data['Experimental Value'],mode='markers',marker=dict(size=5, color=table_data['Experimental Value'], colorscale='Viridis'))])

                fig.update_layout(
                    title="实验数据 3D 可视化",
                    scene=dict(xaxis_title="压力 (MPa)",yaxis_title="温度 (K)",zaxis_title="实验值"))
                st.plotly_chart(fig)
    elif st.session_state.experiment_mode == "机器学习":
        # 🤖 机器学习预测功能
        st.subheader("📈 机器学习预测 - 热导率 & 粘度")

        # 加载模型
        try:
            rf_model_thermal = joblib.load('D:\pythonProject3\streamlit/thermal_conductivity_rf.pkl')
            lin_model_thermal = joblib.load('D:\pythonProject3\streamlit/thermal_conductivity_lin.pkl')
            rf_model_viscosity = joblib.load('D:\pythonProject3\streamlit/viscosity_rf.pkl')
            lin_model_viscosity = joblib.load('D:\pythonProject3\streamlit/viscosity_lin.pkl')
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.stop()

            # 范围输入布局
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("压力范围 (0~40 MPa)")
            min_p = st.number_input("最小值", min_value=0.0, max_value=40.0, value=None, step=5.0, key="ml_min_p")
            max_p = st.number_input("最大值", min_value=0.0, max_value=40.0, value=None, step=5.0, key="ml_max_p")
            step_p = st.number_input("步长", min_value=0.0, max_value=10.0, value=None, step=1.0, key="ml_step_p")

        with col2:
            st.subheader("温度范围 (300~700 K)")
            min_t = st.number_input("最小值", min_value=300.0, max_value=700.0, value=None, step=10.0, key="ml_min_t")
            max_t = st.number_input("最大值", min_value=300.0, max_value=700.0, value=None, step=10.0, key="ml_max_t")
            step_t = st.number_input("步长", min_value=1.0, max_value=50.0, value=None, step=1.0, key="ml_step_t")

        # 确保所有输入框都填了值
        if None in (min_p, max_p, step_p, min_t, max_t, step_t):
            st.warning("⚠️ 请完整填写所有输入框！")
            st.stop()

        # 输入验证
        if not (0.0 <= min_p <= 40.0):
            st.error("❌ 压力最小值必须在 0 到 40 MPa 之间")
            st.stop()

        if not (0.0 <= max_p <= 40.0):
            st.error("❌ 压力最大值必须在 0 到 40 MPa 之间")
            st.stop()

        if min_p >= max_p:
            st.error("❌ 压力最小值必须小于最大值")
            st.stop()

        if not (300.0 <= min_t <= 700.0):
            st.error("❌ 温度最小值必须在 300 到 700 K 之间")
            st.stop()

        if not (300.0 <= max_t <= 700.0):
            st.error("❌ 温度最大值必须在 300 到 700 K 之间")
            st.stop()

        if min_t >= max_t:
            st.error("❌ 温度最小值必须小于最大值")
            st.stop()

        if step_p <= 0 or step_t <= 0:
            st.error("❌ 步长必须大于 0")
            st.stop()

        # 生成参数网格
        pressures = np.arange(min_p, max_p + 1e-9, step_p).round(2)
        temperatures = np.arange(min_t, max_t + 1e-9, step_t).round(1)


        # 预测函数
        def batch_predict(p_arr, t_arr):
            results = []
            total = len(p_arr) * len(t_arr)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, p in enumerate(p_arr):
                for j, t in enumerate(t_arr):
                    # 构造输入数据
                    new_data = pd.DataFrame({'pressure': [p], 'temperature': [t]})

                    # 执行预测
                    try:
                        thermal_rf = rf_model_thermal.predict(new_data)[0]
                        thermal_lin = lin_model_thermal.predict(new_data)[0]
                        viscosity_rf = rf_model_viscosity.predict(new_data)[0]
                        viscosity_lin = lin_model_viscosity.predict(new_data)[0]
                        print(f"预测成功: Thermal_RF={thermal_rf}, Viscosity_RF={viscosity_rf}")
                    except Exception as e:
                        print(f"预测失败: {e}")
                        st.error(f"预测失败 @ P={row['Pressure (MPa)']}MPa, T={row['Temperature (K)']}K: {e}")
                        continue

                    # 保存结果
                    results.append({"Pressure (MPa)": p,"Temperature (K)": t,"Thermal_RF": thermal_rf,"Thermal_LIN": thermal_lin,"Viscosity_RF": viscosity_rf,"Viscosity_LIN": viscosity_lin})

                    # 更新进度
                    progress = (i * len(t_arr) + j + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"进度: {progress * 100:.1f}% 已完成 ({len(results)}/{total} 个数据点)")

            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame(results)

        if st.button("🔍 开始批量预测"):
            with st.spinner("正在生成预测数据，请稍候..."):
                df_results = batch_predict(pressures, temperatures)

            if not df_results.empty:
                st.success(f"成功生成 {len(df_results)} 条预测数据")

                # 显示表格
                st.dataframe(df_results.style.format({"Thermal_RF": "{:.5f}","Thermal_LIN": "{:.5f}","Viscosity_RF": "{:.5f}","Viscosity_LIN": "{:.5f}"}))

                # 下载功能
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载CSV",data=csv,file_name="machine_learning_predictions.csv",mime="text/csv")

            else:
                st.warning("未生成有效预测数据")
        # 添加在机器学习预测部分的适当位置（通常在现有误差计算功能之后）

        if st.button("📊 机器学习预测准确性分析"):
            try:
                # 1. 加载实验数据（分别读取）
                df_viscosity_conditions = pd.read_csv(r'D:\pythonProject3\streamlit\shiyanniandu.csv')
                df_thermal_conditions = pd.read_csv(r'D:\pythonProject3\streamlit\shiyanredaol.csv')

                # 2. 确保数据类型匹配
                df_viscosity_conditions[['pressure', 'temperature']] = df_viscosity_conditions[
                    ['pressure', 'temperature']].astype(float)
                df_thermal_conditions[['pressure', 'temperature']] = df_thermal_conditions[
                    ['pressure', 'temperature']].astype(float)

                # 3. **粘度预测**
                viscosity_predictions = []
                total_viscosity = len(df_viscosity_conditions)  # 总数据点数
                progress_bar_viscosity = st.progress(0)  # 初始化进度条
                status_text_viscosity = st.empty()  # 创建动态文本显示当前进度
                for idx, row in df_viscosity_conditions.iterrows():
                    new_data = pd.DataFrame({'pressure': [row['pressure']], 'temperature': [row['temperature']]})
                    new_data = new_data.astype(float)

                    try:
                        viscosity_rf = rf_model_viscosity.predict(new_data)[0]
                        viscosity_lin = lin_model_viscosity.predict(new_data)[0]
                        exp_value = row.get('nianduexperimentalvalue', None)  # 可能没有实验值

                        # 计算误差
                        error_rf = (abs(viscosity_rf - exp_value) / exp_value) * 100 if exp_value else None
                        error_lin = (abs(viscosity_lin - exp_value) / exp_value) * 100 if exp_value else None

                        viscosity_predictions.append({'Pressure (MPa)': row['pressure'],'Temperature (K)': row['temperature'],'Viscosity_RF': viscosity_rf,'Viscosity_RF_Error (%)': error_rf,'Viscosity_LIN': viscosity_lin,'Viscosity_LIN_Error (%)': error_lin,'Viscosity_Exp': exp_value})

                    except Exception as e:
                        print(f"粘度预测失败（第 {idx} 行）：{e}")

                        # 更新进度条
                    progress = (idx + 1) / total_viscosity
                    progress_bar_viscosity.progress(progress)
                    status_text_viscosity.text(f"粘度预测进度: {progress * 100:.1f}% ({idx + 1}/{total_viscosity})")

                    # 预测完成，清空进度条
                progress_bar_viscosity.empty()
                status_text_viscosity.empty()

                df_viscosity = pd.DataFrame(viscosity_predictions)
                print(f"粘度预测完成：{df_viscosity.shape}")

                # 4. **热导率预测**
                thermal_predictions = []
                total_thermal = len(df_thermal_conditions)
                progress_bar_thermal = st.progress(0)  # 初始化进度条
                status_text_thermal = st.empty()  # 创建动态文本
                for idx, row in df_thermal_conditions.iterrows():
                    new_data = pd.DataFrame({'pressure': [row['pressure']], 'temperature': [row['temperature']]})
                    new_data = new_data.astype(float)

                    try:
                        thermal_rf = rf_model_thermal.predict(new_data)[0]
                        thermal_lin = lin_model_thermal.predict(new_data)[0]
                        exp_value = row.get('redaoexperimentalvalue', None)

                        # 计算误差
                        error_rf = (abs(thermal_rf - exp_value) / exp_value) * 100 if exp_value else None
                        error_lin = (abs(thermal_lin - exp_value) / exp_value) * 100 if exp_value else None

                        thermal_predictions.append({'Pressure (MPa)': row['pressure'],'Temperature (K)': row['temperature'],'Thermal_RF': thermal_rf,'Thermal_RF_Error (%)': error_rf,'Thermal_LIN': thermal_lin,'Thermal_LIN_Error (%)': error_lin,'Thermal_Exp': exp_value})

                    except Exception as e:
                        print(f"热导率预测失败（第 {idx} 行）：{e}")
                    # 更新进度条
                    progress = (idx + 1) / total_thermal
                    progress_bar_thermal.progress(progress)
                    status_text_thermal.text(f"热导率预测进度: {progress * 100:.1f}% ({idx + 1}/{total_thermal})")

                # 预测完成，清空进度条
                progress_bar_thermal.empty()
                status_text_thermal.empty()
                df_thermal = pd.DataFrame(thermal_predictions)
                print(f"热导率预测完成：{df_thermal.shape}")

                # 5. **分别显示两个表**
                st.write("📋 **粘度预测结果（Viscosity）**")
                st.dataframe(df_viscosity)

                st.write("📋 **热导率预测结果（Thermal Conductivity）**")
                st.dataframe(df_thermal)

            except Exception as e:
                st.error(f"分析失败: {e}")

            # # 在显示误差数据框之后添加以下代码
            # plt.rcParams.update({
            #     'font.sans-serif': ['Microsoft YaHei'],  # 使用支持中文和符号的字体
            #     'axes.unicode_minus': True  # 启用Unicode负号显示
            # })
            # # 处理零值（替换为极小正数）
            # df_viscosity['Viscosity_RF_Error (%)'] = df_viscosity['Viscosity_RF_Error (%)'].replace(0, 1e-6)
            # df_viscosity['Viscosity_LIN_Error (%)'] = df_viscosity['Viscosity_LIN_Error (%)'].replace(0, 1e-6)
            # df_thermal['Thermal_RF_Error (%)'] = df_thermal['Thermal_RF_Error (%)'].replace(0, 1e-6)
            # df_thermal['Thermal_LIN_Error (%)'] = df_thermal['Thermal_LIN_Error (%)'].replace(0, 1e-6)
            #
            # # 在误差计算完成后添加以下代码（在显示数据框之后）
            #
            # # 设置热图参数
            # plt.rcParams.update({'font.size': 14})
            # fig, axes = plt.subplots(4, 1, figsize=(18, 30))  # 改为4行1列
            #
            #
            # # 创建辅助函数（添加对数处理）
            # def plot_log_heatmap(data, ax, title, zlabel):
            #     # 转换为矩阵格式
            #     pivot_table = data.pivot_table(index='Temperature (K)',
            #                                    columns='Pressure (MPa)',
            #                                    values=zlabel,
            #                                    aggfunc='mean')
            #
            #     # 使用对数归一化
            #     from matplotlib.colors import LogNorm
            #     norm = LogNorm(vmin=max(1e-6, pivot_table.min().min()),
            #                    vmax=pivot_table.max().max())
            #
            #     sns.heatmap(pivot_table,
            #                 annot=False,
            #                 cmap='coolwarm',
            #                 norm=norm,
            #                 cbar_kws={'label': f'Log({zlabel})',
            #                           'ticks': [0.1, 1, 10, 100]},  # 添加刻度标签
            #                 ax=ax)
            #     ax.set_title(f"Log({title})", fontsize=16, pad=20)
            #     ax.set_xlabel("Pressure (MPa)", fontsize=14)
            #     ax.set_ylabel("Temperature (K)", fontsize=14)
            #     ax.tick_params(axis='both', which='major', labelsize=12)
            #
            #
            # # 绘制粘度误差热图（改为单列布局）
            # plot_log_heatmap(df_viscosity, axes[0],
            #                  "Viscosity RF Error Distribution (%)",
            #                  "Viscosity_RF_Error (%)")
            #
            # plot_log_heatmap(df_viscosity, axes[1],
            #                  "Viscosity LIN Error Distribution (%)",
            #                  "Viscosity_LIN_Error (%)")
            #
            # # 绘制热导率误差热图
            # plot_log_heatmap(df_thermal, axes[2],
            #                  "Thermal RF Error Distribution (%)",
            #                  "Thermal_RF_Error (%)")
            #
            # plot_log_heatmap(df_thermal, axes[3],
            #                  "Thermal LIN Error Distribution (%)",
            #                  "Thermal_LIN_Error (%)")
            #
            # # 调整布局间距
            # plt.subplots_adjust(hspace=0.4)  # 增加垂直间距
            # plt.tight_layout()
            # st.pyplot(fig)