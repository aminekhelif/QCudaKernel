**plot_results.py**

```python
import csv
import matplotlib.pyplot as plt

times = {}
errors = {}

with open("build/results.csv","r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        kernel = row["Kernel"]
        t = float(row["TimeMs"])
        e = float(row["MaxError"])
        times[kernel]=t
        errors[kernel]=e

kernels = list(times.keys())
perf = [times[k] for k in kernels]

plt.figure(figsize=(8,6))
plt.bar(kernels, perf, color='skyblue')
plt.ylabel('Time (ms)')
plt.title('Kernel Performance Comparison')
for i, v in enumerate(perf):
    plt.text(i, v + 0.5, f"{v:.2f} ms", ha='center')
plt.tight_layout()
plt.savefig("performance.png")
print("Performance graph saved as performance.png")