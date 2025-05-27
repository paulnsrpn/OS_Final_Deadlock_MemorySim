from tabulate import tabulate

# --- BANKER'S ALGORITHM SIMULATION ---

def is_safe(processes, available, max_demand, allocation):
    n = len(processes)
    m = len(available)

    # Calculate the Need Matrix
    need = [[max_demand[i][j] - allocation[i][j] for j in range(m)] for i in range(n)]

    finish = [False] * n
    safe_sequence = []
    work = available.copy()

    while len(safe_sequence) < n:
        found = False
        for i in range(n):
            if not finish[i]:
                if all(need[i][j] <= work[j] for j in range(m)):
                    for j in range(m):
                        work[j] += allocation[i][j]
                    finish[i] = True
                    safe_sequence.append(processes[i])
                    found = True
                    break
        if not found:
            return False, []
    return True, safe_sequence

# Define system state
banker_processes = ['P0', 'P1', 'P2', 'P3', 'P4']
available = [5, 3, 2]

max_demand = [
    [7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]
]
allocation = [
    [0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]
]

safe, sequence = is_safe(banker_processes, available, max_demand, allocation)
print("\n--- Banker's Algorithm ---")
if safe:
    print("✅ The system is in a SAFE state.")
    print("🟢 Safe sequence:", ' → '.join(sequence))
else:
    print("❌ The system is in an UNSAFE state (deadlock may occur).")


# --- MEMORY ALLOCATION STRATEGIES ---

def first_fit(blocks, processes):
    allocation = [-1] * len(processes)
    block_status = blocks.copy()
    for i in range(len(processes)):
        for j in range(len(block_status)):
            if block_status[j] >= processes[i]:
                allocation[i] = j
                block_status[j] -= processes[i]
                break
    return allocation

def best_fit(blocks, processes):
    allocation = [-1] * len(processes)
    block_status = blocks.copy()
    for i in range(len(processes)):
        best_index = -1
        for j in range(len(block_status)):
            if block_status[j] >= processes[i]:
                if best_index == -1 or block_status[j] < block_status[best_index]:
                    best_index = j
        if best_index != -1:
            allocation[i] = best_index
            block_status[best_index] -= processes[i]
    return allocation

def worst_fit(blocks, processes):
    allocation = [-1] * len(processes)
    block_status = blocks.copy()
    for i in range(len(processes)):
        worst_index = -1
        for j in range(len(block_status)):
            if block_status[j] >= processes[i]:
                if worst_index == -1 or block_status[j] > block_status[worst_index]:
                    worst_index = j
        if worst_index != -1:
            allocation[i] = worst_index
            block_status[worst_index] -= processes[i]
    return allocation

# Memory allocation simulation
memory_blocks = [100, 500, 200, 300, 600]
processes = [212, 417, 112, 426]

first = first_fit(memory_blocks, processes)
best = best_fit(memory_blocks, processes)
worst = worst_fit(memory_blocks, processes)

# --- CREATE TABLE ---
table = []
for i in range(len(processes)):
    table.append([
        f"P{i}",
        f"{processes[i]} KB",
        f"Block {first[i]}" if first[i] != -1 else "Not Allocated",
        f"Block {best[i]}" if best[i] != -1 else "Not Allocated",
        f"Block {worst[i]}" if worst[i] != -1 else "Not Allocated",
    ])

headers = ["Process", "Size", "First Fit", "Best Fit", "Worst Fit"]

print("\n--- Memory Allocation Table ---")
print(tabulate(table, headers=headers, tablefmt="fancy_grid"))



# DATA VISUALIZATION -----------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Data from simulation
processes = ['P0', 'P1', 'P2', 'P3']
sizes = [212, 417, 112, 426]
allocations = {
    'First Fit': [1, 4, 1, -1],
    'Best Fit': [3, 1, 2, 4],
    'Worst Fit': [4, 1, 4, -1]
}

# Custom colors per strategy
strategy_colors = {
    'First Fit': '#D84040', 
    'Best Fit': '#8E1616',   # blue
    'Worst Fit': '#1D1616'   # orange
}

# Set figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2
x_indexes = list(range(len(processes)))

for i, (strategy, blocks) in enumerate(allocations.items()):
    heights = []
    labels = []

    for block in blocks:
        heights.append(block if block != -1 else 0)
        labels.append(str(block) if block != -1 else 'Not Allocated')

    x_pos = [x + i * bar_width for x in x_indexes]
    
    # Draw bars
    bars = ax.bar(x_pos, heights, width=bar_width, label=strategy, 
                  color=strategy_colors[strategy], alpha=0.8)

    # Add text labels on bars
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, label, 
                ha='center', va='bottom', fontsize=9)

# Title and axis labels
ax.set_title("📊 Memory Allocation Strategy Comparison", fontsize=14, fontweight='bold')
ax.set_xlabel("Processes", fontsize=12)
ax.set_ylabel("Block Assigned (Index)", fontsize=12)
ax.set_xticks([x + bar_width for x in x_indexes])
ax.set_xticklabels(processes, fontsize=11)

# Style
ax.legend(title="Strategy")
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()



# HEATMAP

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Allocation results
allocations = {
    'First Fit': [1, 4, 1, -1],
    'Best Fit': [3, 1, 2, 4],
    'Worst Fit': [4, 1, 4, -1]
}

strategies = list(allocations.keys())
processes = ['P0', 'P1', 'P2', 'P3']

# Convert to numpy array
data = np.array([allocations[strategy] for strategy in strategies])

# Replace -1 with NaN for visualization (meaning Not Allocated)
data = np.where(data == -1, np.nan, data)

# Create heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(data, annot=True, cmap='YlGnBu', cbar=True, linewidths=0.5,
            xticklabels=processes, yticklabels=strategies, fmt='.0f')

plt.title('Memory Block Allocations by Strategy')
plt.xlabel('Processes')
plt.ylabel('Strategies')
plt.tight_layout()
plt.show()


# LINE GRAPH

# Count how many processes were successfully allocated
def count_allocated(allocation):
    return sum(1 for a in allocation if a != -1)

x_labels = ['First Fit', 'Best Fit', 'Worst Fit']
y_values = [
    count_allocated([1, 4, 1, -1]),
    count_allocated([3, 1, 2, 4]),
    count_allocated([4, 1, 4, -1])
]

# Line plot
plt.figure(figsize=(7, 5))
plt.plot(x_labels, y_values, marker='o', linestyle='-', color='purple', linewidth=2)
plt.title('Processes Successfully Allocated')
plt.xlabel('Strategy')
plt.ylabel('Number of Processes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# GRAPHICAL ANALYSIS

import matplotlib.pyplot as plt

# Simulated runtime in milliseconds
runtime_data = {
    'First Fit': 1.2,
    'Best Fit': 1.8,
    'Worst Fit': 2.0
}

plt.figure(figsize=(6, 4))
plt.plot(list(runtime_data.keys()), list(runtime_data.values()), marker='o', color='teal')
plt.title("Runtime of Allocation Strategies")
plt.xlabel("Strategy")
plt.ylabel("Runtime (ms)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# FRAGMENTATION CHART

# Simulated used memory
fragmentation_data = {
    'First Fit': 212 + 417 + 112,  # Process 3 was not allocated
    'Best Fit': 212 + 417 + 112 + 426,
    'Worst Fit': 212 + 417 + 112,  # Process 3 not allocated
}

total_memory = 1700
wasted_memory = {k: total_memory - v for k, v in fragmentation_data.items()}

# Plot
labels = list(fragmentation_data.keys())
used = list(fragmentation_data.values())
wasted = list(wasted_memory.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, used, label='Used Memory', color='mediumseagreen')
plt.bar(labels, wasted, bottom=used, label='Fragmentation', color='lightcoral')
plt.title('Memory Utilization and Fragmentation')
plt.ylabel('Memory (KB)')
plt.legend()
plt.tight_layout()
plt.show()


# 

