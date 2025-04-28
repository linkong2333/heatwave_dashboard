import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json

# ---------------------- 数据读取与预处理 ----------------------
data_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
df = pd.read_csv(data_path)

# 清理列名（去除空格）
df.columns = df.columns.str.strip()
print("数据各列情况:")
print(df.columns)
print(df.head())

# 提取经纬度（如果有 .geo 字段）
if '.geo' in df.columns:
    def extract_lon_lat(geo_str):
        try:
            geo_obj = json.loads(geo_str)
            coords = geo_obj.get('coordinates', [None, None])
            return pd.Series({"longitude": coords[0], "latitude": coords[1]})
        except Exception as e:
            return pd.Series({"longitude": None, "latitude": None})

    coords_df = df['.geo'].apply(extract_lon_lat)
    df = pd.concat([df, coords_df], axis=1)
    print("成功提取 longitude 和 latitude 列。")

# 检查关键字段
required_cols = ['LST_Celsius', 'Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH', 'longitude', 'latitude']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print("缺失字段:", missing)
else:
    print("所有关键字段均存在。")

# 删除关键字段缺失的样本
df = df.dropna(subset=required_cols)

# ---------------------- 处理 LST_Celsius 异常值 ----------------------
# 利用 IQR 方法处理异常值
Q1 = df['LST_Celsius'].quantile(0.25)
Q3 = df['LST_Celsius'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"LST_Celsius 异常值检测: 下界 = {lower_bound}, 上界 = {upper_bound}")

# 仅保留 LST_Celsius 在合理范围内的样本
df = df[(df['LST_Celsius'] >= lower_bound) & (df['LST_Celsius'] <= upper_bound)]
print("处理异常值后样本数量:", len(df))

# ---------------------- 特征与目标变量 ----------------------
# 使用 Albedo, BuildingDensity, GreenCover, NDVI, RH 作为特征，预测 LST_Celsius
X = df[['Albedo', 'BuildingDensity', 'GreenCover', 'NDVI', 'RH']]
y = df['LST_Celsius']

# 划分训练集和测试集
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, random_state=42
)

# ---------------------- 随机森林回归模型训练 ----------------------
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# 预测并评估
y_pred = rf_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("测试集 R^2: {:.2f}".format(r2))
print("测试集 MAE: {:.2f}".format(mae))

# 将预测结果存入测试集 DataFrame
df_test = df_test.copy()
df_test['LST_Predicted'] = y_pred

# 输出测试集真实值与预测值的统计数据
print("Test set actual LST_Celsius statistics:")
print(df_test['LST_Celsius'].describe())
print("\nTest set predicted LST_Celsius statistics:")
print(df_test['LST_Predicted'].describe())

# ---------------------- 空间可视化 ----------------------
# 利用经纬度构造 GeoDataFrame
geometry = [Point(xy) for xy in zip(df_test['longitude'], df_test['latitude'])]
gdf = gpd.GeoDataFrame(df_test, geometry=geometry, crs="EPSG:4326")

# 可视化预测结果
plt.figure(figsize=(10, 8))
gdf.plot(column='LST_Predicted', cmap='coolwarm', legend=True, markersize=50)
plt.title("Predicted LST_Celsius (Random Forest Regression)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 可视化实际 LST_Celsius
plt.figure(figsize=(10, 8))
gdf.plot(column='LST_Celsius', cmap='coolwarm', legend=True, markersize=50)
plt.title("Actual LST_Celsius")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# ---------------------- 模型保存 ----------------------
# joblib.dump(rf_regressor, "rf_model_regressor.pkl")



import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
import os
import json

# ---------------------- 第一步：读取每个区的人口表格 ----------------------
root_path = "人口"
districts = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]

population_data = []

for district in districts:
    try:
        path_m01 = os.path.join(root_path, district, "Table M01.csv")
        path_m02 = os.path.join(root_path, district, "Table M02.csv")

        # 跳过前4行
        m01 = pd.read_csv(path_m01, skiprows=4)
        m02 = pd.read_csv(path_m02, skiprows=4)

        m01.columns = m01.columns.str.strip()
        m02.columns = m02.columns.str.strip()

        # 选需要的列
        m01 = m01[['年', 'Unnamed: 1', 'Unnamed: 2', '2024']].dropna()
        m02 = m02[['年', 'Unnamed: 1', '2024.3']].dropna()

        m01_total = m01[m01['Unnamed: 1'] == '總計']

        # 提取人口数据
        children_pop = m01_total[m01_total['Unnamed: 2'] == '0 - 14歲']['2024'].values[0]
        elderly_pop = m01_total[m01_total['Unnamed: 2'] == '65歲及以上']['2024'].values[0]
        total_pop = m01_total[m01_total['Unnamed: 2'] == '總計']['2024'].values[0]

        # 计算比例
        children_ratio = children_pop / total_pop
        elderly_ratio = elderly_pop / total_pop

        # 提取住戶每月入息中位數
        m02_total = m02[m02['Unnamed: 1'] == '合計']
        income_median = m02_total['2024.3'].values[0]

        population_data.append({
            'District_CN': district,
            'Elderly_Ratio': elderly_ratio,
            'Children_Ratio': children_ratio,
            'Income_Median': income_median
        })
    except Exception as e:
        print(f"读取 {district} 出错: {e}")

# 整合成DataFrame
pop_df = pd.DataFrame(population_data)
print("人口脆弱性数据整理完成：")
print(pop_df)

# 【加一步】中文区名映射为英文区名
district_mapping = {
    '中西区': 'Central and Western District',
    '东区': 'Eastern District',
    '南区': 'Southern District',
    '湾仔': 'Wan Chai District',
    '离岛': 'Islands District',
    '油尖旺': 'Yau Tsim Mong District',
    '观塘': 'Kwun Tong District',
    '深水涉': 'Sham Shui Po District',  # 注意你的原中文是"深水涉"，这里映射为深水埗区的英文
    '九龙城': 'Kowloon City District',
    '黄大仙': 'Wong Tai Sin District',
    '葵青': 'Kwai Tsing District',
    '荃湾': 'Tsuen Wan District',
    '屯门': 'Tuen Mun District',
    '沙田': 'Sha Tin District',
    '西贡': 'Sai Kung District',
    '北区': 'North District',
    '大甫': 'Tai Po District',
    '西贡': 'Sai Kung District',
}

pop_df['District'] = pop_df['District_CN'].map(district_mapping)

# 删除中文列
pop_df = pop_df.drop(columns=['District_CN'])

# ---------------------- 第二步：处理热浪预测点数据 ----------------------
data_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
df = pd.read_csv(data_path)

# 清理列名
df.columns = df.columns.str.strip()


# 提取经纬度
def extract_lon_lat(geo_str):
    try:
        geo_obj = json.loads(geo_str)
        coords = geo_obj.get('coordinates', [None, None])
        return pd.Series({"longitude": coords[0], "latitude": coords[1]})
    except Exception:
        return pd.Series({"longitude": None, "latitude": None})


if '.geo' in df.columns:
    coords_df = df['.geo'].apply(extract_lon_lat)
    df = pd.concat([df, coords_df], axis=1)
    print("成功提取经纬度！")

geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# ---------------------- 第三步：加载区界并空间连接 ----------------------
districts_gdf = gpd.read_file("DistrictBoundary_SHP/DCD.shp")
districts_gdf = districts_gdf.to_crs("EPSG:4326")

# 用英文区名
districts_gdf = districts_gdf.rename(columns={'NAME_EN': 'District'})
districts_gdf = districts_gdf[['District', 'geometry']]

# 空间连接
gdf_with_district = gpd.sjoin(gdf, districts_gdf, how="left", predicate="within")

# 合并人口脆弱性指标
gdf_with_district = gdf_with_district.merge(pop_df, on='District', how='left')

print("空间连接与属性合并完成：")
print(gdf_with_district.head())

# ---------------------- 第四步：计算脆弱性指数 ----------------------
scaler = MinMaxScaler()

# 归一化
gdf_with_district['LST_norm'] = scaler.fit_transform(gdf_with_district[['LST_Celsius']])
gdf_with_district['Elderly_norm'] = scaler.fit_transform(gdf_with_district[['Elderly_Ratio']])
gdf_with_district['Children_norm'] = scaler.fit_transform(gdf_with_district[['Children_Ratio']])
gdf_with_district['Income_Median_norm'] = 1 - scaler.fit_transform(gdf_with_district[['Income_Median']])  # 收入越低越脆弱

# 综合脆弱性指数计算
gdf_with_district['Vulnerability_Index'] = (
        gdf_with_district['LST_norm'] * 0.5 +
        gdf_with_district['Elderly_norm'] * 0.2 +
        gdf_with_district['Children_norm'] * 0.2 +
        gdf_with_district['Income_Median_norm'] * 0.1
)

# ---------------------- 第五步：绘制脆弱性地图 ----------------------
# 绘制预测点脆弱性
fig, ax = plt.subplots(figsize=(12, 10))

# 第一层：绘制预测点脆弱性
gdf_with_district.plot(
    column='Vulnerability_Index',
    cmap='RdYlGn_r',  # 分色更明显
    legend=True,
    markersize=40,
    alpha=0.8,  # 稍微透明一点，方便叠加边界
    legend_kwds={'label': "Vulnerability Index", 'shrink': 0.6},
    ax=ax
)

# 第二层：叠加行政区边界（只画线框）
districts_gdf.boundary.plot(
    ax=ax,
    edgecolor='black',
    linewidth=1.0
)

# 美化设置
ax.set_title("Urban Heatwave Vulnerability Index with Administrative Boundaries", fontsize=16)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.5)

plt.show()

import matplotlib.cm as cm
import matplotlib.colors as colors

# 计算每个区的平均脆弱性
district_mean = gdf_with_district.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False)

# 归一化脆弱性指数用于着色（0-1之间）
norm = colors.Normalize(vmin=district_mean.min(), vmax=district_mean.max())
cmap = cm.get_cmap('RdYlGn_r')  # 注意是反转（r），高值红色，低值绿色
bar_colors = [cmap(norm(value)) for value in district_mean]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 8))

district_mean.plot(
    kind='bar',
    color=bar_colors,
    ax=ax
)

# 美化
ax.set_title("Average Urban Heatwave Vulnerability Index by District", fontsize=16)
ax.set_xlabel("District")
ax.set_ylabel("Average Vulnerability Index")
ax.set_ylim(0.4, 0.7)  # ✅ 收缩y轴，让趋势更明显
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors

# 重新计算每个点属于哪个District，并保留Vulnerability_Index
district_vi = gdf_with_district[['District', 'Vulnerability_Index']].dropna()

# 计算每个区的均值用于排序
district_order = district_vi.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False).index

# 归一化均值用于着色
mean_values = district_vi.groupby('District')['Vulnerability_Index'].mean()
norm = colors.Normalize(vmin=mean_values.min(), vmax=mean_values.max())
cmap = cm.get_cmap('RdYlGn_r')

# 生成每个区对应的颜色
palette = {district: cmap(norm(value)) for district, value in mean_values.items()}

# 绘制箱线图
plt.figure(figsize=(14, 8))
sns.boxplot(
    data=district_vi,
    x='District',
    y='Vulnerability_Index',
    order=district_order,
    palette=palette,
    showfliers=False
)

# 美化
plt.title("Distribution of Urban Heatwave Vulnerability Index by District", fontsize=16)
plt.xlabel("District")
plt.ylabel("Vulnerability Index")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

# ------------------ 引入包 ------------------
import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from branca.colormap import linear
st.set_page_config(page_title="Urban Heatwave Vulnerability Dashboard", layout="wide")
# ------------------ 读取数据（加缓存） ------------------
@st.cache_data
def load_data():
    points = gpd.read_file("heatwave_vulnerability_points.gpkg")  # ✅ 这里改成你的同级目录
    boundaries = gpd.read_file("district_boundaries.shp")
    return points, boundaries

gdf_with_district, districts_gdf = load_data()

# ------------------ 页面设置 ------------------


st.title("Urban Heatwave Vulnerability Index Dashboard")
st.markdown("Visualize heatwave risk distribution and social vulnerability across Hong Kong districts.")

# ------------------ 侧边栏选择 ------------------
option = st.sidebar.selectbox(
    "Select Visualization",
    ("Map of Vulnerability Index", "Bar Chart of District Average", "Box Plot of Vulnerability Distribution")
)

# ------------------ 地图 ------------------
if option == "Map of Vulnerability Index":
    st.header("Heatwave Vulnerability Map")

    m = folium.Map(location=[22.3964, 114.1095], zoom_start=11, tiles='cartodb positron')

    # 颜色映射器
    colormap = linear.RdYlGn_11.scale(
        gdf_with_district['Vulnerability_Index'].min(),
        gdf_with_district['Vulnerability_Index'].max()
    )
    colormap.colors = list(reversed(colormap.colors))  # 手动反转色带
    colormap = colormap.to_step(18)

    for idx, row in gdf_with_district.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            fill=True,
            fill_opacity=0.7,
            fill_color=colormap(row['Vulnerability_Index']),
            color=None
        ).add_to(m)

    folium.GeoJson(
        districts_gdf.geometry,
        style_function=lambda feature: {'color': 'black', 'weight': 1, 'fillOpacity': 0}
    ).add_to(m)

    colormap.caption = 'Urban Heatwave Vulnerability Index'
    colormap.add_to(m)

    st_data = st_folium(
        m,
        width=1000,
        height=700,
        key="heatwave-map",
        returned_objects=[],
    )

# ------------------ 柱状图 ------------------
elif option == "Bar Chart of District Average":
    st.header("District-wise Average Vulnerability Index")

    district_mean = gdf_with_district.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False)

    norm = colors.Normalize(vmin=district_mean.min(), vmax=district_mean.max())
    cmap = cm.get_cmap('RdYlGn_r')
    bar_colors = [cmap(norm(value)) for value in district_mean]

    fig, ax = plt.subplots(figsize=(14, 8))
    district_mean.plot(kind='bar', color=bar_colors, ax=ax)
    ax.set_title("Average Urban Heatwave Vulnerability Index by District", fontsize=16)
    ax.set_xlabel("District")
    ax.set_ylabel("Average Vulnerability Index")
    ax.set_ylim(0.4, 0.75)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    # 下载按钮
    csv = district_mean.reset_index().to_csv(index=False)
    st.download_button(
        label="Download District Vulnerability CSV",
        data=csv,
        file_name='district_vulnerability.csv',
        mime='text/csv',
    )

# ------------------ 箱线图 ------------------
elif option == "Box Plot of Vulnerability Distribution":
    st.header("Distribution of Vulnerability Index in Districts")

    district_vi = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
    district_order = district_vi.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False).index

    mean_values = district_vi.groupby('District')['Vulnerability_Index'].mean()
    norm = colors.Normalize(vmin=mean_values.min(), vmax=mean_values.max())
    cmap = cm.get_cmap('RdYlGn_r')
    palette = {district: cmap(norm(value)) for district, value in mean_values.items()}

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=district_vi,
        x='District',
        y='Vulnerability_Index',
        order=district_order,
        palette=palette,
        showfliers=False
    )

    ax.set_title("Distribution of Urban Heatwave Vulnerability Index by District", fontsize=16)
    ax.set_xlabel("District")
    ax.set_ylabel("Vulnerability Index")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)