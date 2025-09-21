# ==========================================
# # Mihan Mohagheghzadeh
# GELP Project- September 2025
# ==========================================


# ==========================================
# Fake Data Simulation + Plotting
# Design: 3 (Word Order) * 3 (Object Realization)
# ==========================================

#==========================================
# Libraries and Packages
#==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------
# 1. Factors
# --------------------------
word_orders = ["Verb-final", "Verb-medial", "Verb-initial"]
object_types = ["Overt NP", "Clitic", "Null"]

# number of simulated participants
n_participants = 30  

 # how many trials per condition per participant
trials_per_condition = 10 

# --------------------------
# 2. fake expected latencies in ms
# --------------------------
true_means = {
    ("Verb-final", "Overt NP"): 1350,
    ("Verb-final", "Clitic"): 1420,
    ("Verb-final", "Null"): 1500,
    ("Verb-medial", "Overt NP"): 1450,
    ("Verb-medial", "Clitic"): 1520,
    ("Verb-medial", "Null"): 1600,
    ("Verb-initial", "Overt NP"): 1550,
    ("Verb-initial", "Clitic"): 1620,
    ("Verb-initial", "Null"): 1700,
}

# standard deviation of noise 
sigma = 80

# --------------------------
# 3. Generating fake participant-level data
# --------------------------
rows = []
for pid in range(1, n_participants + 1):
    for wo in word_orders:
        for obj in object_types:
            mean_latency = true_means[(wo, obj)]

            # simulating latencies with Gaussian noise
            latencies = np.random.normal(loc = mean_latency, scale = sigma, size = trials_per_condition)
            for latency in latencies:
                rows.append({"Participant": pid, "WordOrder": wo, "Object": obj, "Latency": latency})

df = pd.DataFrame(rows)

# --------------------------
# 4. means and SEs per condition
# --------------------------
agg = df.groupby(["WordOrder", "Object"]).agg(
    MeanLatency=("Latency", "mean"),
    SE=("Latency", lambda x: np.std(x) / np.sqrt(len(x)))
).reset_index()

print(agg)

# --------------------------
# 5. Bar plot with error bars
# --------------------------
plt.figure(figsize = (8, 6))

sns.barplot(
    data = agg,
    x = "WordOrder",
    y = "MeanLatency",
    hue = "Object",
    palette = "Set2",
    capsize = 0.1,
    errcolor = "black",
    errwidth = 1.5
)
plt.title("Speech-onset Latencies by Word Order * Object Realization")
plt.ylabel("Mean Latency (ms)")
plt.xlabel("Word Order")
plt.legend(title = "Object Realization")
plt.tight_layout()
plt.show()

# --------------------------
# 6. Line Plot for Interaction
# --------------------------
plt.figure(figsize = (8, 6))
sns.pointplot(
    data = agg,
    x = "WordOrder",
    y = "MeanLatency",
    hue = "Object",
    palette = "Set2",
    markers = "o",
    capsize = 0.1,
    errwidth = 1.5
)
plt.title("Interaction: Word Order * Object Realization")
plt.ylabel("Mean Latency (ms)")
plt.xlabel("Word Order")
plt.legend(title = "Object Realization")
plt.tight_layout()
plt.show()
