import os
from PIL import Image
from datasets import Dataset,load_from_disk
from tqdm import tqdm



def QA_template(sftquestion, solution, img_path):
    return {
        "messages": [
            {"role": "user", "content": f"<image>{sftquestion}"},
            {"role": "assistant", "content": solution}
        ],
        "images": [img_path]
    }

def get_bbox_list_from_file(label_path):
    # 读取并解析 bboxes
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            coords = line.strip().split()
            if len(coords) == 4:
                try:
                    bboxes.append([float(c) for c in coords])
                except:
                    continue

    if not bboxes:
        return None

    # 缩放 bbox
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height

    scaled_bboxes = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        scaled = [
            int(min(max(round(x1 * scale_x), 0), target_size[0] - 1)),
            int(min(max(round(y1 * scale_y), 0), target_size[1] - 1)),
            int(min(max(round(x2 * scale_x), 0), target_size[0] - 1)),
            int(min(max(round(y2 * scale_y), 0), target_size[1] - 1))
        ]
        scaled_bboxes.append(scaled)

    return scaled_bboxes


if __name__ == '__main__':
    # 初始化数据结构 - 只存储图像路径，不存储图像对象
    features = {
        "problem": [],
        "solution": [],
        "image_path": [],
        "image":[],
        "target":[]
    }

    images_root = "your_dataset/train" #eg:"/IMAGENET/20class_25/train"

    output_dataset_path = "save_path" #eg:f"/dataset/IMN"
    save_path = f"{output_dataset_path}/train"

    object_classes = ['analog clock', 'backpack', 'ballpoint', 'Band Aid', 'barbell', 'barber chair', 'beer bottle', 'beer glass', 'binoculars', 'bolo tie', 'bookcase', 'bottlecap', 'brassiere', 'broom', 'bucket', 'buckle', 'candle', 'can opener', 'carton', 'cellular telephone']


    # # 先收集所有文件路径
    # 处理数据
    for classname in tqdm(object_classes, desc="处理图像"):
        classfolder=f"{images_root}/{classname}"
        for imgname in os.listdir(classfolder):
            # 加载原始图像
            img_path=f"{classfolder}/{imgname}"
            image = Image.open(img_path).convert('RGB')
            original_width, original_height = image.size  # 获取原始尺寸

            # 设置目标尺寸
            target_size = (1000, 1000)

            # 调整图像尺寸
            resized_image = image.resize(target_size, Image.LANCZOS)

            result_str = "{" + ", ".join(f"'{cls_name}':prob_{i}" for i, cls_name in enumerate(object_classes)) + "}"

            answer_str ="{" + ", ".join( f"'{cls_name}':1.0" if cls_name==classname else f"'{cls_name}':0.0" for cls_name in object_classes) + "}"
            question = f"""Analyze the image to explain how the visual elements support the object classification probability distribution {answer_str}, concisely in under 100 words. Describe key objects, shapes, textures, and contextual details that contribute to the inferred object category, highlighting their connection to the classification probabilities. Avoid directly stating the object names or probabilities; instead, focus on the visual cues and their implications."""
            answer = f"<answer>{answer_str}</answer>"

            features["image_path"].append(img_path)
            features["problem"].append(question)
            features["solution"].append(answer)
            features["image"].append(resized_image)  # 存储调整后的图像
            features["target"].append(classname)

    # 创建完整数据集 - 只存储路径
    full_dataset = Dataset.from_dict(features)

    # 保存整个数据集
    os.makedirs(save_path, exist_ok=True)
    full_dataset.save_to_disk(save_path)
    # 在 save_to_disk 之后写一下数据集划分
    with open(f"{output_dataset_path}/dataset_dict.json", "w") as f:
        f.write('{"splits": ["train"]}')

    print(f"数据集已保存到: {save_path}")

    # 统计信息
    total = len(features["image_path"])
    print("\n数据集摘要:")
    print(f"总样本数: {total}")
