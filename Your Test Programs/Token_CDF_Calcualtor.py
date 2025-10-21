import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =============================
# 🔢 INSERT YOUR DATA BELOW
# Token counts for each category
# =============================

# 💭 Thinking model
thinking_input_tokens = np.array([
    365, 789, 1107, 1451, 1954, 2486, 2569, 2736, 2853, 2795, 2757, 2653, 2597, 2605, 2707, 2779, 3010, 3696, 3616, 3521, 3455, 3240, 2356, 2324, 2360, 2334, 2428, 2702, 2902, 2960, 3027, 3096, 2918, 2789, 2626, 2533, 2704, 2606, 2607, 2591, 2750, 2538, 2657, 2746, 2857, 2815, 2641, 2617, 2529, 2543, 2675, 2775, 2930, 2942, 3049, 3190, 3193, 3219, 3183, 2930, 2705, 2532, 2355, 2147, 2129, 2070, 2139, 2099, 2461, 2605, 2805, 2993, 3021, 2805, 2718, 2648, 2631, 2554, 2594, 2678, 2783, 2876, 2955, 3019, 3054, 3040, 2926, 2784, 2650, 2461, 2408, 2517, 2812, 3061, 3301, 3537, 3487, 3416, 3117, 2960, 2810, 2668, 2684, 2714, 2692, 2668, 2647, 2685, 2716
])
thinking_output_tokens = np.array([
    82, 96, 90, 85, 88, 108, 105, 101, 132, 127, 110, 97, 124, 104, 121, 101, 99, 119, 104, 102, 100, 90, 114, 96, 117, 135, 109, 110, 100, 99, 127, 134, 97, 117, 107, 97, 123, 114, 140, 117, 96, 93, 104, 98, 88, 89, 101, 109, 105, 125, 115, 129, 109, 92, 115, 111, 114, 113, 103, 107, 105, 145, 118, 93, 103, 73, 101, 98, 131, 110, 106, 113, 113, 103, 108, 83, 115, 103, 104, 99, 119, 105, 106, 140, 115, 109, 129, 99, 105, 84, 132, 127, 94, 113, 119, 97, 88, 107, 110, 110, 90, 96, 91, 96, 105, 99, 114, 123, 110
])

# 🚫 Without-thinking model
no_thinking_input_tokens = np.array([
287, 514, 727, 989, 1207, 1529, 1683, 1770, 1906, 1931, 2044, 2026, 1900, 1947, 1952, 1964, 1890, 1938, 1925, 1828, 1852, 1909, 2046, 2020, 2185, 2351, 2315, 2285, 2220, 2244, 2189, 2163, 2149, 2141, 2122, 2138, 2096, 2008, 1987, 1955, 1960, 1885, 1844, 1794, 1759, 1782, 1693, 1673, 1638, 1612, 1566, 1479, 1491, 1490, 1551, 1699, 1741, 1795, 1885, 1991, 1975, 1877, 1951, 1937, 1824, 1655, 1627, 1611, 1472, 1448, 1467, 1518, 1518, 1559, 1734, 1922, 1990, 2156, 2316, 2408, 2488, 2355, 2501, 2503, 2379, 2308, 2181, 2302, 2078, 1906, 1824, 1707, 1612, 1414, 1410, 1365, 1360, 1415, 1439, 1547, 1711, 1782, 1900, 1950, 1966, 1908
])
no_thinking_output_tokens = np.array([
    3, 7, 9, 7, 7, 7, 9, 9, 9, 7, 11, 9, 9, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 9, 9, 9, 7, 9, 9, 9, 9, 9, 9, 11, 7, 9, 13, 9, 9, 9, 7, 9, 9, 9, 9, 9, 4, 9, 7, 7, 6, 9, 9, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 7, 9, 7, 9, 7, 9, 11, 9, 11, 11, 11, 9, 7, 11, 11, 9, 11, 11, 9, 10, 9, 9, 11, 100, 9, 9, 11, 9, 7, 7, 7, 7, 9, 9, 5, 7, 7, 9, 9, 9, 7
])

# =============================
# 📈 CDF Functions
# =============================

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    sorted_data = np.insert(sorted_data, 0, 0)
    cdf = np.insert(cdf, 0, 0.0)
    return sorted_data, cdf

def smooth_cdf(x, y, points=200):
    f = interp1d(x, y, kind='linear')
    x_smooth = np.linspace(x.min(), x.max(), points)
    y_smooth = f(x_smooth)
    return x_smooth, y_smooth

# =============================
# 📊 Compute, Smooth, and Plot
# =============================

x1, y1 = compute_cdf(thinking_input_tokens)
x2, y2 = compute_cdf(thinking_output_tokens)
x3, y3 = compute_cdf(no_thinking_input_tokens)
x4, y4 = compute_cdf(no_thinking_output_tokens)

x1_smooth, y1_smooth = smooth_cdf(x1, y1)
x2_smooth, y2_smooth = smooth_cdf(x2, y2)
x3_smooth, y3_smooth = smooth_cdf(x3, y3)
x4_smooth, y4_smooth = smooth_cdf(x4, y4)

plt.figure(figsize=(8, 6))
plt.plot(x1_smooth, y1_smooth, label='Thinking - Input Tokens', linewidth=2)
plt.plot(x2_smooth, y2_smooth, label='Thinking - Output Tokens', linewidth=2)
plt.plot(x3_smooth, y3_smooth, label='No Thinking - Input Tokens', linewidth=2)
plt.plot(x4_smooth, y4_smooth, label='No Thinking - Output Tokens', linewidth=2)

plt.xlabel('Token Count')
plt.ylabel('CDF ([0, 1])')
plt.title('CDF of Input/Output Token Counts')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('token_cdf_plot.png')
