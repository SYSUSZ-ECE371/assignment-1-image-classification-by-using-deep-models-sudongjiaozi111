import os
import shutil
import random
import argparse

def prepare_dataset(dataset_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    # 获取所有类别目录
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()

    # 写入 classes.txt
    classes_file = os.path.join(dataset_dir, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"已生成 {classes_file}")

    train_list = []
    val_list = []
    # 为每个类别划分数据并移动
    for idx, cls in enumerate(classes):
        src_cls_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(src_cls_dir)
                  if os.path.isfile(os.path.join(src_cls_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # 创建目标目录
        train_cls_dir = os.path.join(dataset_dir, 'train', cls)
        val_cls_dir = os.path.join(dataset_dir, 'val', cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        # 移动并记录文件列表
        for img in train_imgs:
            src = os.path.join(src_cls_dir, img)
            dst = os.path.join(train_cls_dir, img)
            shutil.move(src, dst)
            rel_path = os.path.join('train', cls, img)
            train_list.append(f"{rel_path} {idx}")

        for img in val_imgs:
            src = os.path.join(src_cls_dir, img)
            dst = os.path.join(val_cls_dir, img)
            shutil.move(src, dst)
            rel_path = os.path.join('val', cls, img)
            val_list.append(f"{rel_path} {idx}")

    # 写入 train.txt
    train_txt = os.path.join(dataset_dir, 'train.txt')
    with open(train_txt, 'w', encoding='utf-8') as f:
        for line in train_list:
            f.write(line + "\n")
    print(f"已生成 {train_txt}")

    # 写入 val.txt
    val_txt = os.path.join(dataset_dir, 'val.txt')
    with open(val_txt, 'w', encoding='utf-8') as f:
        for line in val_list:
            f.write(line + "\n")
    print(f"已生成 {val_txt}")

    # 移除原始类别目录
    for cls in classes:
        src_cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(src_cls_dir):
            shutil.rmtree(src_cls_dir)
            print(f"已移除原始目录 {src_cls_dir}")

    print("数据集已按 ImageNet 格式准备完毕。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备 flower_dataset: 按 ImageNet 格式分割训练/验证集')
    parser.add_argument('--dataset_dir', type=str, default='flower_dataset', help='数据集目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例，默认 0.8')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认 42')
    args = parser.parse_args()

    prepare_dataset(args.dataset_dir, args.train_ratio, args.seed)