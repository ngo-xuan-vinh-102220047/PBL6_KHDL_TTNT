import os
import matplotlib.pyplot as plt

data_test_dir = "dataset/Test"
data_train_dir = "dataset/Train"
test_classes = os.listdir(data_test_dir)
train_classes = os.listdir(data_train_dir)
counts = []

print("Test data:")
for cls in test_classes:
    cls_dir = os.path.join(data_test_dir, cls)
    if os.path.isdir(cls_dir):
        num_files = len(os.listdir(cls_dir))
        counts.append(num_files)
        print(f"{cls}: {num_files} files")
print('---------------------')
print("Train data:")
for cls in train_classes:
    cls_dir = os.path.join(data_train_dir, cls)
    if os.path.isdir(cls_dir):
        num_files = len(os.listdir(cls_dir))
        counts.append(num_files)
        print(f"{cls}: {num_files} files")

# Vẽ biểu đồ
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].bar(test_classes, counts[:len(test_classes)])
axs[0].set_xticklabels(test_classes, rotation=45)
axs[0].set_ylabel("Số lượng file")
axs[0].set_title("Số lượng file của từng class trong data test")

axs[1].bar(train_classes, counts[len(test_classes):])
axs[1].set_xticklabels(train_classes, rotation=45)
axs[1].set_ylabel("Số lượng file")
axs[1].set_title("Số lượng file của từng class trong data train")

plt.tight_layout()
plt.show()