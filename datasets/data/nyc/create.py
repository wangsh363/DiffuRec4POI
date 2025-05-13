import pandas as pd
import argparse
import os


def generate_item_entities(csv_path, output_path):
    """
    根据 CSV 文件生成 item_entities.dict 文件。

    参数：
        csv_path (str): 输入 CSV 文件路径（包含 POI_id 列）
        output_path (str): 输出 item_entities.dict 文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件 {csv_path} 不存在")

    # 读取 CSV 文件，仅加载 POI_id 列
    try:
        df = pd.read_csv(csv_path, usecols=['POI_id'])
    except KeyError:
        raise ValueError("CSV 文件缺少 'POI_id' 列")
    except Exception as e:
        raise ValueError(f"读取 CSV 文件失败: {e}")

    # 提取唯一的 POI_id 并转换为整数
    try:
        poi_ids = df['POI_id'].dropna().astype(int).unique()
    except ValueError:
        raise ValueError("POI_id 包含非整数值")

    if len(poi_ids) == 0:
        raise ValueError("CSV 文件中没有有效的 POI_id")

    # 生成连续 ID（从 1 开始）
    smap = {i : poi_id for i, poi_id in enumerate(sorted(poi_ids))}

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入 item_entities.dict
    try:
        with open(output_path, 'w') as f:
            for continuous_id, raw_poi_id in smap.items():
                f.write(f"{continuous_id} {raw_poi_id}\n")
    except Exception as e:
        raise IOError(f"写入 {output_path} 失败: {e}")

    print(f"成功生成 {output_path}，包含 {len(smap)} 个 POI ID")
    print(f"连续 ID 范围: [1, {max(smap.keys())}]")
    print(f"原始 POI ID 示例: {list(smap.values())[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 CSV 文件生成 item_entities.dict")
    parser.add_argument('--csv_path', type=str, required=True, help="输入 CSV 文件路径（例如 NYC_train.csv）")
    parser.add_argument('--output_path', type=str, default='item_entities.dict',
                        help="输出 item_entities.dict 文件路径")
    args = parser.parse_args()

    try:
        generate_item_entities(args.csv_path, args.output_path)
    except Exception as e:
        print(f"错误: {e}")