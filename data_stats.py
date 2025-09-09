import os
import matplotlib.pyplot as plt

data_dir = "data"
# Đếm số lượng file NormalVideos
normal_count = len(os.listdir(os.path.join(data_dir, "NormalVideos")))
unnormal_count = len(os.listdir(os.path.join(data_dir, "UnnormalVideos")))

print(f"Số file NormalVideos: {normal_count}")
print(f"Số file UnnormalVideos: {unnormal_count}")

# Trực quan hóa tương quan
labels = ["NormalVideos", "UnnormalVideos"]
counts = [normal_count, unnormal_count]

plt.figure(figsize=(6,6))
bars = plt.bar(labels, counts, color=["blue", "orange"])
plt.ylabel("Số lượng file")
plt.title("So sánh số lượng file: Normal vs Unnormal")
plt.tight_layout()

# Hiển thị số lượng file trên từng cột
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count), 
             ha='center', va='bottom', fontsize=12)

plt.show()