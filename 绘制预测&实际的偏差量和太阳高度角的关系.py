import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 手动实现简易光伏预测模型 (替代 pvlib)
# ==========================================
def calculate_solar_position_and_radiation(lat, lon, dates, timezone_offset=9.5):
    """
    简易太阳位置与辐射计算 (基于 Meinel & Meinel 模型)
    适用于无 pvlib 库环境
    """
    doy = dates.dayofyear
    lat_rad = np.radians(lat)
    
    # 1. 赤纬角 (Declination)
    delta = 23.45 * np.sin(np.radians(360/365 * (doy - 81)))
    delta_rad = np.radians(delta)
    
    # 2. 时差 (Equation of Time)
    B = np.radians(360/365 * (doy - 81))
    eot = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    
    # 3. 太阳时计算
    lstm = 15 * timezone_offset
    tc = 4 * (lon - lstm) + eot 
    local_time_minutes = dates.hour * 60 + dates.minute + dates.second / 60
    solar_time_minutes = local_time_minutes + tc
    
    # 4. 时角 (Hour Angle)
    omega = (solar_time_minutes / 4) - 180
    omega_rad = np.radians(omega)
    
    # 5. 太阳高度角 (Elevation)
    sin_alpha = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(omega_rad)
    alpha = np.degrees(np.arcsin(np.clip(sin_alpha, -1, 1)))
    
    # 6. DNI 计算 (Meinel 模型 - 适合沙漠干燥环境)
    sin_alpha_clamped = np.maximum(sin_alpha, 0.01)
    am = 1 / sin_alpha_clamped
    dni = 1353 * 0.7**(am**0.678)
    dni = np.where(alpha > 0, dni, 0)
    
    return dni

# ==========================================
# 1. 配置与数据加载
# ==========================================
DATA_FILE = 'cleaned_data.csv'
LAT = -23.76
LON = 133.87
P_RATED = 26.5
SYSTEM_EFFICIENCY = 0.85 # 系统综合效率

# 选取稳定晴天 (5天平均)
PLOT_START = '2023-02-09'
PLOT_END = '2023-02-14'

# 读取数据
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 筛选并重采样
mask = (df.index >= PLOT_START) & (df.index < PLOT_END)
df_subset = df.loc[mask].copy()
df_subset = df_subset.resample('15min').mean()

# ==========================================
# 2. 计算理论值
# ==========================================
dni_est = calculate_solar_position_and_radiation(LAT, LON, df_subset.index)
# 双轴跟踪假设：板面辐射(POA) ≈ DNI
df_subset['Predicted'] = P_RATED * (dni_est / 1000) * SYSTEM_EFFICIENCY
df_subset['Actual'] = df_subset['Active_Power']

# ==========================================
# 3. 核心步骤：计算日均分布 (5:00-19:00)
# ==========================================
# 筛选白天
df_daytime = df_subset.between_time('07:00', '19:00').copy()

# 【关键】按“时间”分组求均值 -> 得到单日的平均形态
daily_profile = df_daytime.groupby(df_daytime.index.time).mean()
# ... (数据加载与处理代码同上) ...


# -----------【在此处添加缺失的代码】-----------
# 计算偏差 (Predicted - Actual)
# 注意：虽然变量名叫 Abs_Diff，但根据你的图例 'Predicted - Actual' 来看，
# 这里应该保留正负号（残差），而不是取绝对值。
daily_profile['Abs_Diff'] = daily_profile['Predicted'] - daily_profile['Actual']
# ==========================================
# 1. 复用之前的核心计算逻辑 (保持一致性)
# ==========================================
# (为了代码简洁，这里假设你已经运行了之前的步骤，df_daytime 已经包含了 'Predicted' 和 'Actual')
# 如果是独立运行，请确保 df_daytime 已经被计算出来（包含 2023-02-09 到 2023-02-13 的数据）

# 这里我们需要重新计算一下 Solar Elevation (Alpha) 用于绘图
def get_solar_elevation(lat, lon, dates, timezone_offset=9.5):
    doy = dates.dayofyear
    lat_rad = np.radians(lat)
    delta = 23.45 * np.sin(np.radians(360/365 * (doy - 81)))
    delta_rad = np.radians(delta)
    B = np.radians(360/365 * (doy - 81))
    eot = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    lstm = 15 * timezone_offset
    tc = 4 * (lon - lstm) + eot
    local_time_minutes = dates.hour * 60 + dates.minute + dates.second / 60
    solar_time_minutes = local_time_minutes + tc
    omega = (solar_time_minutes / 4) - 180
    omega_rad = np.radians(omega)
    sin_alpha = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(omega_rad)
    alpha = np.degrees(np.arcsin(np.clip(sin_alpha, -1, 1)))
    return alpha

# ==========================================
# 2. 准备绘图数据
# ==========================================
# 重新加载数据 (确保环境独立)
DATA_FILE = 'cleaned_data.csv'
LAT = -23.76
LON = 133.87
TZ_OFFSET = 9.5
P_RATED = 26.5
SYSTEM_EFFICIENCY = 0.85 
PLOT_START = '2023-02-09'
PLOT_END = '2023-02-14'

df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
mask = (df.index >= PLOT_START) & (df.index < PLOT_END)
df_subset = df.loc[mask].copy()
df_subset = df_subset.resample('15min').mean()

# 重新计算预测值和高度角
alpha = get_solar_elevation(LAT, LON, df_subset.index, TZ_OFFSET)
sin_alpha_clamped = np.maximum(np.sin(np.radians(alpha)), 0.01)
am = 1 / sin_alpha_clamped
dni = 1353 * 0.7**(am**0.678)
dni = np.where(alpha > 0, dni, 0)

df_subset['Predicted'] = P_RATED * (dni / 1000) * SYSTEM_EFFICIENCY
df_subset['Actual'] = df_subset['Active_Power']
df_subset['Elevation'] = alpha
df_subset['Abs_Diff'] = df_subset['Predicted'] - df_subset['Actual']

# 筛选白天有效时段
df_daytime = df_subset.between_time('06:00', '20:00').copy()

# 区分上午和下午 (用于诊断东西向遮挡)
# 定义中午为 12:45 (当地太阳时中午大约是 12:45 而不是 12:00，因为经度 133.87 vs 时区中心 142.5)
# 简单起见，按时刻 13:00 分割
df_am = df_daytime[df_daytime.index.hour < 13]
df_pm = df_daytime[df_daytime.index.hour >= 13]

# ==========================================
# 3. 绘制：偏差 vs 高度角
# ==========================================
plt.figure(figsize=(10, 6), dpi=100)

# 绘制上午数据 (AM) - 蓝色
plt.scatter(df_am['Elevation'], df_am['Abs_Diff'], 
            color='#1f77b4', alpha=0.6, s=30, label='Morning (AM)')

# 绘制下午数据 (PM) - 橙色
plt.scatter(df_pm['Elevation'], df_pm['Abs_Diff'], 
            color='#ff7f0e', alpha=0.6, s=30, marker='x', label='Afternoon (PM)')

# 添加辅助线
plt.axhline(0, color='black', linewidth=1)
plt.axvline(10, color='gray', linestyle='--', label='10° Elevation Threshold') # 假设的遮挡阈值

# 装饰
plt.title('Figure 4: Prediction Deviation vs. Solar Elevation Angle\n(Diagnosing Shading & Horizon Effects)', fontsize=12, fontweight='bold')
plt.xlabel('Solar Elevation Angle (Degrees)', fontsize=11)
plt.ylabel('Absolute Deviation (Predicted - Actual) [kW]', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# 限制一下X轴，只看 0度到90度
plt.xlim(0, 90)

plt.tight_layout()
plt.savefig('deviation_vs_elevation.png')
print("图表已保存！请观察低角度（左侧）是否有显著的偏差翘尾现象。")
plt.show()