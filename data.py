# Project: Earthquake Load Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

df = pd.read_excel('/Users/jamesbaxter/Documents/University/3rd Year/Earthquake Load Data.xlsx',
                     sheet_name='Processed Data', usecols='EF:EI')
df2 = pd.read_excel('/Users/jamesbaxter/Documents/University/3rd Year/Final Results.xlsx')

# Extract the data from the excel file.
EW = df['EW.1']
NS = df['NS']
UD = df['UD']
Time = df['period']

# Print the column names of the excel file.
#print(df2.columns)
# Time_90_Days = df2['90 Days Corroded']
# Stress_90_days = df2['Avg Stress']
# DAMAGET_90_days = df2['Avg DAMAGET']
# PE_90_days = df2['Avg PE']
# MAX_S_90_days = df2['MAX S']

# Time_180_Days = df2['TIME']
# Stress_180_days = df2['AVG STRESS']
# DAMAGET_180_days = df2['Avg DAMAGET.1']
# PE_180_days = df2['Avg PE.1']
# MAX_S_180_days = df2['MAX S.1']

# Time_270_Days = df2['Time.1']
# Stress_270_days = df2['AVG STRESS.1']
# DAMAGET_270_days = df2['Avg DAMAGET.2']
# PE_270_days = df2['Avg PE.2']
# MAX_S_270_days = df2['MAX S.2']

# Time_Bending = df2['Time.2']
# Stress_Bending = df2['Avg Stress']
# DAMAGET_Bending = df2['Avg DAMAGET.3']
# PE_Bending = df2['Avg PE.3']
# MAX_S_Bending = df2['MAX S.3']

# plt.plot(Time_90_Days, Stress_90_days, label='90 Days Corroded')
# plt.plot(Time_180_Days, Stress_180_days, label='180 Days Corroded')
# plt.plot(Time_270_Days, Stress_270_days, label='270 Days Corroded')
# plt.plot(Time_Bending, Stress_Bending, label='Bending')
# plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
# plt.ylabel('Stress (Pa)', fontsize=12, fontweight='bold')
# plt.title('Stress vs Time', fontsize=14, fontweight='bold')
# plt.show()


# Use matplotlib animation to plot, EW against Time, NS against Time, UD against Time for 300 seconds of Time data.
# It will use FuncAnimation to update the plot every 10ms.
# The plots will all be plotted on the same axis, not separate plots.
# The linewidth of the plots will be 0.5.
# Include a timer in the bottom right corner of the plot.

upper_bound = max([max(EW), max(NS), max(UD)])
lower_bound = min([min(EW), min(NS), min(UD)])

fig = plt.figure()
ax = plt.axes(xlim=(0, 300), ylim=(lower_bound, upper_bound))
line, = ax.plot([], [], lw=0.5)
line2, = ax.plot([], [], lw=0.5)
line3, = ax.plot([], [], lw=0.5)

def init():
    line.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line, line2, line3


def animate(i):
    x = Time[:i]
    y = EW[:i]
    y2 = NS[:i]
    y3 = UD[:i]
    line.set_data(x, y)
    line2.set_data(x, y2)
    line3.set_data(x, y3)
    return line, line2, line3


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30000, interval=10, blit=True)
plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
plt.ylabel('Acceleration (m/$s^2$)', fontsize=12, fontweight='bold')
plt.title('Earthquake Acceleration Data', fontsize=14, fontweight='bold')
plt.legend(['EW', 'NS', 'UD'], loc='upper left', frameon=True, fancybox=True, shadow=True, framealpha=1)
plt.show()










