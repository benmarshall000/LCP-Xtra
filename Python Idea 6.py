import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from io import StringIO

# Logistic function from original code
def logistic(x): 
    return 1 / (1 + np.exp(-x)) 

def compute_V(P_L, w_diff, t_after, c_after, beta_w, beta_t, beta_c, beta_p, gamma):
    return gamma + beta_w * (w_diff/1000) - beta_t * t_after - beta_c * c_after + beta_p * np.log(P_L)

# Load the CSV data
csv_data = """Town,Region,Average_Salary_GBP,Nearest_Major_City,Train_Time_Minutes,City_Average_Salary_GBP,Salary_Difference_GBP
Blackpool,Lancashire,29500,Manchester,75,38500,9000
Burnley,Lancashire,31200,Manchester,65,38500,7300
Accrington,Lancashire,30800,Manchester,70,38500,7700
Nelson,Lancashire,30500,Manchester,80,38500,8000
Colne,Lancashire,30200,Manchester,85,38500,8300
Darwen,Lancashire,31800,Bolton,45,34000,2200
Blackburn,Lancashire,32000,Manchester,55,38500,6500
Preston,Lancashire,33800,Manchester,50,38500,4700
Lancaster,Lancashire,33200,Manchester,90,38500,5300
Morecambe,Lancashire,31500,Manchester,95,38500,7000
Barrow-in-Furness,Cumbria,32200,Manchester,120,38500,6300
Workington,Cumbria,30800,Newcastle,110,37200,6400
Whitehaven,Cumbria,31000,Newcastle,115,37200,6200
Carlisle,Cumbria,33000,Newcastle,90,37200,4200
Penrith,Cumbria,32800,Newcastle,85,37200,4400
Hartlepool,County Durham,30500,Newcastle,80,37200,6700
Darlington,County Durham,33500,Newcastle,45,37200,3700
Bishop Auckland,County Durham,31200,Newcastle,70,37200,6000
Durham,County Durham,34500,Newcastle,40,37200,2700
Consett,County Durham,31800,Newcastle,60,37200,5400
Hexham,Northumberland,33800,Newcastle,50,37200,3400
Blyth,Northumberland,31500,Newcastle,35,37200,5700
Cramlington,Northumberland,34200,Newcastle,25,37200,3000
Ashington,Northumberland,31200,Newcastle,40,37200,6000
Wansbeck,Northumberland,30800,Newcastle,45,37200,6400
Rhyl,Denbighshire,29800,Manchester,85,38500,8700
Prestatyn,Denbighshire,30200,Manchester,90,38500,8300
Llandudno,Conwy,32500,Manchester,75,38500,6000
Bangor,Gwynedd,31800,Manchester,95,38500,6700
Caernarfon,Gwynedd,31200,Manchester,100,38500,7300
Pwllheli,Gwynedd,30500,Manchester,120,38500,8000
Aberystwyth,Ceredigion,32000,Birmingham,110,36800,4800
Cardigan,Pembrokeshire,30800,Cardiff,105,37500,6700
Haverfordwest,Pembrokeshire,32200,Cardiff,95,37500,5300
Milford Haven,Pembrokeshire,31500,Cardiff,100,37500,6000
Fishguard,Pembrokeshire,30200,Cardiff,115,37500,7300
Newtown,Powys,31000,Birmingham,90,36800,5800
Welshpool,Powys,31800,Birmingham,85,36800,5000
Machynlleth,Powys,30500,Birmingham,100,36800,6300
Brecon,Powys,32000,Cardiff,80,37500,5500
Merthyr Tydfil,Rhondda Cynon Taf,31200,Cardiff,55,37500,6300
Aberdare,Rhondda Cynon Taf,31800,Cardiff,50,37500,5700
Mountain Ash,Rhondda Cynon Taf,30800,Cardiff,60,37500,6700
Pontypridd,Rhondda Cynon Taf,33000,Cardiff,35,37500,4500
Caerphilly,Caerphilly,32500,Cardiff,30,37500,5000
Bargoed,Caerphilly,31000,Cardiff,45,37500,6500
Blackwood,Caerphilly,31500,Cardiff,40,37500,6000
Tredegar,Blaenau Gwent,30200,Cardiff,70,37500,7300
Ebbw Vale,Blaenau Gwent,30500,Cardiff,65,37500,7000
Abergavenny,Monmouthshire,33200,Cardiff,55,37500,4300"""

df = pd.read_csv(StringIO(csv_data))

# Parameters from original model
P_s = 20000       # small population
S = 0.6           # working-age share
c_after = 10      # travel cost (£ per period)

# Default model parameters
default_beta_w = 0.28
default_beta_t = 0.0375
default_beta_c = -0.0005
default_beta_p = 0.3
default_gamma = -6
default_w_diff = 25.0

# City populations (estimates for the model)
city_populations = {
    'Manchester': 800000,
    'Leeds': 750000,
    'Sheffield': 550000,
    'Liverpool': 650000,
    'Newcastle': 450000,
    'York': 200000,
    'Preston': 150000,
    'Hull': 280000
}

def calculate_commuter_proportion_and_new_salary(row, time_reduction, beta_w, beta_t, beta_c, beta_p, gamma):
    """Calculate the proportion of people commuting and new average salary"""
    city = row['Nearest_Major_City']
    P_L = city_populations.get(city, 500000)  # Default if city not found
    
    # Apply time reduction
    adjusted_time = row['Train_Time_Minutes'] * (1 - time_reduction/100)
    
    # Calculate utility
    V = compute_V(P_L, row['Salary_Difference_GBP'], adjusted_time, c_after, 
                  beta_w, beta_t, beta_c, beta_p, gamma)
    
    # Calculate proportion who commute
    p_commute = logistic(V)
    
    # Calculate new average salary
    new_avg_salary = (p_commute * row['City_Average_Salary_GBP'] + 
                     (1 - p_commute) * row['Average_Salary_GBP'])
    
    return p_commute, new_avg_salary

# Set up the plot with more space for sliders
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
plt.subplots_adjust(bottom=0.35, left=0.1, right=0.95)

# Create sliders with more space
axcolor = '#F39200'
slider_height = 0.02
slider_spacing = 0.025
start_y = 0.28

# Sliders from original file
ax_w_diff = plt.axes([0.15, start_y, 0.7, slider_height], facecolor=axcolor)
ax_bw = plt.axes([0.15, start_y - slider_spacing, 0.7, slider_height], facecolor=axcolor)
ax_bt = plt.axes([0.15, start_y - 2*slider_spacing, 0.7, slider_height], facecolor=axcolor)
ax_bc = plt.axes([0.15, start_y - 3*slider_spacing, 0.7, slider_height], facecolor=axcolor)
ax_bp = plt.axes([0.15, start_y - 4*slider_spacing, 0.7, slider_height], facecolor=axcolor)
ax_gamma = plt.axes([0.15, start_y - 5*slider_spacing, 0.7, slider_height], facecolor=axcolor)
ax_time_reduction = plt.axes([0.15, start_y - 6*slider_spacing, 0.7, slider_height], facecolor=axcolor)

# Create sliders
s_w_diff = Slider(ax_w_diff, 'Avg wage difference (£1k)', -5, 30, valinit=default_w_diff, track_color = '#00A3C7',color = '#E93F6F')
s_bw = Slider(ax_bw, 'Higher wage desire (β_w)', 0.001, 0.7, valinit=default_beta_w, track_color = '#00A3C7',color = '#E93F6F')
s_bt = Slider(ax_bt, 'Commute time intolerance (β_t)', 0.001, 0.06, valinit=default_beta_t, track_color = '#00A3C7',color = '#E93F6F')
s_bc = Slider(ax_bc, 'Travel price tolerance (β_c)', -0.1, -0.0005, valinit=default_beta_c, track_color = '#00A3C7',color = '#E93F6F')
s_bp = Slider(ax_bp, 'Population preference (β_p)', 0.1, 0.8, valinit=default_beta_p, track_color = '#00A3C7',color = '#E93F6F')
s_gamma = Slider(ax_gamma, 'Constant (γ)', -10.0, 3.0, valinit=default_gamma, track_color = '#00A3C7',color = '#E93F6F')
s_time_reduction = Slider(ax_time_reduction, 'Travel time reduction (%)', 0, 50, valinit=0, valfmt='%d%%', track_color = '#00A3C7',color = '#E93F6F')

def update_plots():
    """Update both plots based on current slider values"""
    # Clear the axes properly
    ax1.cla()
    ax2.cla()
    
    # Get current slider values
    w_diff = s_w_diff.val
    beta_w = s_bw.val
    beta_t = s_bt.val
    beta_c = s_bc.val
    beta_p = s_bp.val
    gamma = s_gamma.val
    time_reduction = s_time_reduction.val
    
    # Left plot: Original logistic model (like Python Idea.py)
    P_L = 800000  # Default large city population
    t_vals = np.linspace(0, 240, 241)
    V_vals = compute_V(P_L, w_diff*1000, t_vals, c_after, beta_w, beta_t, beta_c, beta_p, gamma)
    p_vals = logistic(V_vals)
    
    ax1.plot(t_vals, p_vals, 'b-', linewidth=2,color = '#E93F6F')
    ax1.set_xlabel('Travel time (minutes)')
    ax1.set_ylabel('Predicted proportion commuting')
    ax1.set_ylim(0, 1)
    ax1.set_title('Logistic Model: Commuting Probability vs Travel Time')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Dumbbell chart for towns
    results = []
    for _, row in df.iterrows():
        p_commute, new_salary = calculate_commuter_proportion_and_new_salary(
            row, time_reduction, beta_w, beta_t, beta_c, beta_p, gamma)
        results.append({
            'Town': row['Town'],
            'Region': row['Region'],
            'Original_Salary': row['Average_Salary_GBP'],
            'New_Salary': new_salary,
            'Commute_Proportion': p_commute,
            'Salary_Change': new_salary - row['Average_Salary_GBP']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Original_Salary')
    n_towns = len(results_df)
    
    # Create exactly one dumbbell per town
    for i in range(n_towns):
        orig = results_df['Original_Salary'].iloc[i]
        new = results_df['New_Salary'].iloc[i]
        change = results_df['Salary_Change'].iloc[i]
        
        # Color code based on salary change
        line_color = '#F39200' if change > 0 else 'red' if change < 0 else 'gray'
        line_alpha = min(abs(change) / 2000, 1.0)
        
        # Plot exactly one line and two points per town
        ax2.plot([orig, new], [i, i], color=line_color, alpha=max(0.3, line_alpha), linewidth=2)
        ax2.scatter(orig, i, color='#E93F6F', s=40, alpha=0.7)
        ax2.scatter(new, i, color='#00A3C7', s=40, alpha=0.8)
    
    # Add legend manually with dummy plots
    ax2.scatter([], [], color='#E93F6F', s=40, alpha=0.7, label='Original Salary')
    ax2.scatter([], [], color='#00A3C7', s=40, alpha=0.8, label='With Commuting')
    
    # Show labels for selected towns
    step = max(1, n_towns // 20)
    show_indices = list(range(0, n_towns, 1))  #1 was step
    if (n_towns - 1) not in show_indices:
        show_indices.append(n_towns - 1)
    
    ax2.set_yticks(show_indices)
    ax2.set_yticklabels([f"{results_df['Town'].iloc[i]} ({results_df['Commute_Proportion'].iloc[i]:.1%})" 
                        for i in show_indices], fontsize=7)
    ax2.set_xlabel('Average Salary (£)')
    
    title_suffix = f" (Travel Time -{time_reduction:.0f}%)" if time_reduction > 0 else ""
    ax2.set_title(f'Salary Impact of City Commuting{title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    # Add statistics
    avg_change = results_df['Salary_Change'].mean()
    max_change = results_df['Salary_Change'].max()
    ax2.text(0.02, 0.98, f'Avg change: £{avg_change:.0f}\nMax change: £{max_change:.0f}', 
             transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='#F39200', alpha=0.2))
    
    #fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)

def slider_update(val):
    update_plots()
    plt.draw()

# Connect all sliders to update function
for slider in [s_w_diff, s_bw, s_bt, s_bc, s_bp, s_gamma, s_time_reduction]:
    slider.on_changed(slider_update)

# Initial plot
update_plots()

plt.show()

# Print some sample results
print("\nSample Results with Current Settings:")
print("-" * 80)
sample_results = []
for i, row in df.head(8).iterrows():
    p_commute, new_salary = calculate_commuter_proportion_and_new_salary(
        row, 0, default_beta_w, default_beta_t, default_beta_c, default_beta_p, default_gamma)
    sample_results.append({
        'Town': row['Town'][:12],  # Truncate for formatting
        'Travel_Time': f"{row['Train_Time_Minutes']}min",
        'Original_Salary': f"£{row['Average_Salary_GBP']:,}",
        'New_Salary': f"£{int(new_salary):,}",
        'Commute_%': f"{p_commute:.1%}",
        'Change': f"£{int(new_salary - row['Average_Salary_GBP']):,}"
    })

sample_df = pd.DataFrame(sample_results)
print(sample_df.to_string(index=False))

