import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Panel 1: accuracy progression ----
labels = ["Week 2\n(easy:\nplant vs forest)",
          "Week 3\n(hard:\nNO2 only)",
          "Week 4\n(hard:\nNO2+SO2)",
          "Week 5\n(hard:\nNO2+SO2+VIIRS)"]
# Week 5 measured via train_3channel.py: 79.2% hard-only -- an exact
# tie with Week 4, i.e. VIIRS MaxFRP did not improve accuracy here
# (see WEEK5_LOG.txt for the full breakdown).
acc = [91.2, 77.1, 79.2, 79.2]
colors = ["#7f8c8d", "#c0392b", "#27ae60", "#2980b9"]
bars = ax1.bar(labels, acc, color=colors)
ax1.axhline(50, ls="--", color="gray", label="chance (50%)")
ax1.set_ylabel("Test accuracy (%)"); ax1.set_ylim(0, 100)
ax1.set_title("Detection accuracy across the study")
for b, a in zip(bars, acc):
    ax1.text(b.get_x()+b.get_width()/2, a+1.5, f"{a}%", ha="center", fontweight="bold")
ax1.legend()

# ---- Panel 2: city vs steel false-alarm shift (NO2 -> NO2+SO2) ----
sources = ["Delhi", "Mumbai", "Bangalore", "Kolkata",   # cities
           "Bhilai", "Jamshedpur"]                        # steel
week3 = [0.43, 0.53, 0.55, 0.33, 0.55, 0.58]
week4 = [0.11, 0.30, 0.40, 0.32, 0.59, 0.54]
x = np.arange(len(sources)); w = 0.35
ax2.bar(x-w/2, week3, w, label="NO2 only (W3)", color="#c0392b")
ax2.bar(x+w/2, week4, w, label="NO2+SO2 (W4)", color="#27ae60")
ax2.axhline(0.5, ls="--", color="gray", label="decision threshold")
ax2.set_xticks(x); ax2.set_xticklabels(sources, rotation=30)
ax2.set_ylabel("avg P(plant)  — lower = better")
ax2.set_title("False-alarm rate: cities drop, steel persists")
ax2.legend()
# shade the steel region
ax2.axvspan(3.5, 5.5, alpha=0.08, color="orange")
ax2.text(4.5, 0.7, "STEEL\n(still confused)", ha="center", fontsize=9, color="#b9770e")

plt.tight_layout()
plt.savefig("data/summary_weeks2to4.png", dpi=130, bbox_inches="tight")
print("Saved data/summary_weeks2to4.png")