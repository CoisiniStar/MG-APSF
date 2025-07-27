# check_dataset.py
import json

def check_dataset_size(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"文件: {file_path}")
    print(f"总条目数: {len(data)}")
    
    # 统计各标签的数量
    label_counts = {}
    for item in data:
        label = item.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("各标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  标签 {label}: {count} 条")
    print("-" * 40)

# 检查三个数据集
check_dataset_size("datasets/Split_output_QID_train.json")
check_dataset_size("datasets/Split_output_QID_val.json")
check_dataset_size("datasets/Split_output_QID_test.json")