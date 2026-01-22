import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
df_daytime = df_subset.between_time('06:00', '20:00').copy()

# 【关键】按“时间”分组求均值 -> 得到单日的平均形态
daily_profile = df_daytime.groupby(df_daytime.index.time).mean()
# ... (数据加载与处理代码同上) ...


# -----------【在此处添加缺失的代码】-----------
# 计算偏差 (Predicted - Actual)
# 注意：虽然变量名叫 Abs_Diff，但根据你的图例 'Predicted - Actual' 来看，
# 这里应该保留正负号（残差），而不是取绝对值。
daily_profile['Abs_Diff'] = daily_profile['Predicted'] - daily_profile['Actual']
# ==========================================
# 绘图 (图2：绝对偏差柱状图)
# ==========================================
plt.figure(figsize=(10, 6), dpi=100)

# 使用 dummy_date 转换时间轴
dummy_date = pd.to_datetime('2023-01-01')
plot_times = [dummy_date + pd.Timedelta(hours=t.hour, minutes=t.minute) for t in daily_profile.index]

# 绘制绿色柱状图
# width 设置为 0.008 (约等于11分钟宽，略小于15分钟间隔，留出缝隙)
plt.bar(plot_times, daily_profile['Abs_Diff'], width=0.008, 
        color='#2ca02c', alpha=0.8, label='Absolute Deviation (Predicted - Actual)')

# 添加 0 轴基准线
plt.axhline(0, color='black', linewidth=1)

plt.title('Figure 2: Average Absolute Deviation (Predicted - Actual)\nPeriod: 2023-02-09 to 2023-02-13 (06:00-20:00)', fontsize=12)
plt.xlabel('Time of Day', fontsize=11)
plt.ylabel('Deviation (kW)', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 格式化 X 轴
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()
plt.savefig('average_absolute_deviation.png')
plt.show()