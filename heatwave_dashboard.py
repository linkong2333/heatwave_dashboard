import streamlit as st
st.set_page_config(page_title="Urban Heatwave Vulnerability Dashboard", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_training_data():
    data_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json

# ---------------------- 训练数据处理与模型训练部分 ----------------------
# 保持原有封装，不动



# ---------------------- 脆弱性数据处理与空间连接部分，封装为函数 ----------------------
@st.cache_data
def process_vulnerability_data():
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from sklearn.preprocessing import MinMaxScaler
    import os
    import json
    # 1. 人口数据
    root_path = "人口"
    districts = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    population_data = []
    for district in districts:
        try:
            path_m01 = os.path.join(root_path, district, "Table M01.csv")
            path_m02 = os.path.join(root_path, district, "Table M02.csv")
            m01 = pd.read_csv(path_m01, skiprows=4)
            m02 = pd.read_csv(path_m02, skiprows=4)
            m01.columns = m01.columns.str.strip()
            m02.columns = m02.columns.str.strip()
            m01 = m01[['年', 'Unnamed: 1', 'Unnamed: 2', '2024']].dropna()
            m02 = m02[['年', 'Unnamed: 1', '2024.3']].dropna()
            m01_total = m01[m01['Unnamed: 1'] == '總計']
            children_pop = m01_total[m01_total['Unnamed: 2'] == '0 - 14歲']['2024'].values[0]
            elderly_pop = m01_total[m01_total['Unnamed: 2'] == '65歲及以上']['2024'].values[0]
            total_pop = m01_total[m01_total['Unnamed: 2'] == '總計']['2024'].values[0]
            children_ratio = children_pop / total_pop
            elderly_ratio = elderly_pop / total_pop
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
    pop_df = pd.DataFrame(population_data)
    district_mapping = {
        '中西区': 'Central and Western District',
        '东区': 'Eastern District',
        '南区': 'Southern District',
        '湾仔': 'Wan Chai District',
        '离岛': 'Islands District',
        '油尖旺': 'Yau Tsim Mong District',
        '观塘': 'Kwun Tong District',
        '深水涉': 'Sham Shui Po District',
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
    pop_df = pop_df.drop(columns=['District_CN'])
    # 2. 热浪预测点数据
    data_path = r"Heatwave_Training_Data_With_RoadDensity (4).csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
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
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    # 3. 行政区界
    districts_gdf = gpd.read_file("DistrictBoundary_SHP/DCD.shp")
    districts_gdf = districts_gdf.to_crs("EPSG:4326")
    districts_gdf = districts_gdf.rename(columns={'NAME_EN': 'District'})
    districts_gdf = districts_gdf[['District', 'geometry']]
    # 4. 空间连接
    gdf_with_district = gpd.sjoin(gdf, districts_gdf, how="left", predicate="within")
    gdf_with_district = gdf_with_district.merge(pop_df, on='District', how='left')
    # 5. 脆弱性指数
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    gdf_with_district['LST_norm'] = scaler.fit_transform(gdf_with_district[['LST_Celsius']])
    gdf_with_district['Elderly_norm'] = scaler.fit_transform(gdf_with_district[['Elderly_Ratio']])
    gdf_with_district['Children_norm'] = scaler.fit_transform(gdf_with_district[['Children_Ratio']])
    gdf_with_district['Income_Median_norm'] = 1 - scaler.fit_transform(gdf_with_district[['Income_Median']])
    gdf_with_district['Vulnerability_Index'] = (
        gdf_with_district['LST_norm'] * 0.5 +
        gdf_with_district['Elderly_norm'] * 0.2 +
        gdf_with_district['Children_norm'] * 0.2 +
        gdf_with_district['Income_Median_norm'] * 0.1
    )
    return gdf_with_district, districts_gdf

# ------------------ 绘图函数封装 ------------------
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from branca.colormap import linear
import seaborn as sns

def draw_map(gdf_with_district, districts_gdf):
    st.header("Heatwave Vulnerability Map")
    m = folium.Map(location=[22.3964, 114.1095], zoom_start=11, tiles='cartodb positron')
    colormap = linear.RdYlGn_11.scale(
        gdf_with_district['Vulnerability_Index'].min(),
        gdf_with_district['Vulnerability_Index'].max()
    )
    colormap.colors = list(reversed(colormap.colors))
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
    st_folium(
        m,
        width=1000,
        height=700,
        key="heatwave-map",
        returned_objects=[],
    )

def draw_bar_chart(gdf_with_district):
    st.header("District-wise Average Vulnerability Index")
    district_mean = gdf_with_district.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False)
    norm = colors.Normalize(vmin=district_mean.min(), vmax=district_mean.max())
    cmap = cm.get_cmap('RdYlGn_r')
    bar_colors = [cmap(norm(value)) for value in district_mean]
    import matplotlib.pyplot as plt
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

def draw_box_plot(gdf_with_district):
    st.header("Distribution of Vulnerability Index in Districts")
    district_vi = gdf_with_district[['District', 'Vulnerability_Index']].dropna()
    district_order = district_vi.groupby('District')['Vulnerability_Index'].mean().sort_values(ascending=False).index
    mean_values = district_vi.groupby('District')['Vulnerability_Index'].mean()
    norm = colors.Normalize(vmin=mean_values.min(), vmax=mean_values.max())
    cmap = cm.get_cmap('RdYlGn_r')
    palette = {district: cmap(norm(value)) for district, value in mean_values.items()}
    import matplotlib.pyplot as plt
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

# ------------------ Streamlit 主流程 ------------------
st.title("Urban Heatwave Vulnerability Index Dashboard")
st.markdown("Visualize heatwave risk distribution and social vulnerability across Hong Kong districts.")

# 数据加载
df = load_training_data()
gdf_with_district, districts_gdf = process_vulnerability_data()

# 侧边栏选择
option = st.sidebar.selectbox(
    "Select Visualization",
    ("Map of Vulnerability Index", "Bar Chart of District Average", "Box Plot of Vulnerability Distribution")
)

if option == "Map of Vulnerability Index":
    draw_map(gdf_with_district, districts_gdf)
elif option == "Bar Chart of District Average":
    draw_bar_chart(gdf_with_district)
elif option == "Box Plot of Vulnerability Distribution":
    draw_box_plot(gdf_with_district)