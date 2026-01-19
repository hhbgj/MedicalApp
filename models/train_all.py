"""
一键训练所有医学图像识别模型
按顺序训练：肺炎 → 乳腺癌 → 脑肿瘤 → 疟疾
"""

import subprocess
import sys
import os
import time

print("=" * 70)
print("🏥 医学图像AI - 批量模型训练程序")
print("=" * 70)

BASE_DIR = 'C:/MedicalAI'
MODELS_DIR = f'{BASE_DIR}/models'

# 训练脚本列表（按推荐顺序）
TRAINING_SCRIPTS = [
    {
        'script': 'train_pneumonia.py',
        'name': '🫁 肺炎检测',
        'dataset': 'chest_xray',
        'estimated_time': '20-40分钟'
    },
    {
        'script': 'train_breast.py',
        'name': '🎀 乳腺癌检测',
        'dataset': 'breast_ultrasound',
        'estimated_time': '10-20分钟'
    },
    {
        'script': 'train_brain.py',
        'name': '🧠 脑肿瘤检测',
        'dataset': 'brain_tumor',
        'estimated_time': '5-15分钟'
    },
    {
        'script': 'train_malaria.py',
        'name': '🦟 疟疾检测',
        'dataset': 'malaria',
        'estimated_time': '30-60分钟'
    }
]

def check_dataset(dataset_name):
    """检查数据集是否存在"""
    dataset_path = f'{BASE_DIR}/datasets/{dataset_name}'
    return os.path.exists(dataset_path)

def run_training(script_info):
    """运行单个训练脚本"""
    script_path = f'{MODELS_DIR}/{script_info["script"]}'
    
    if not os.path.exists(script_path):
        return 'missing_script'
    
    if not check_dataset(script_info['dataset']):
        return 'missing_dataset'
    
    print(f"\n{'='*60}")
    print(f"开始训练: {script_info['name']}")
    print(f"预计时间: {script_info['estimated_time']}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=MODELS_DIR,
            check=True
        )
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n✅ {script_info['name']} 训练完成! 用时: {elapsed:.1f}分钟")
        return 'success'
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_info['name']} 训练失败: {e}")
        return 'failed'
    except Exception as e:
        print(f"\n❌ {script_info['name']} 出错: {e}")
        return 'error'

def main():
    """主函数"""
    print("\n📋 数据集检查:")
    
    available_trainings = []
    missing_datasets = []
    
    for script_info in TRAINING_SCRIPTS:
        has_dataset = check_dataset(script_info['dataset'])
        status = '✅ 已就绪' if has_dataset else '❌ 缺少数据集'
        print(f"   {script_info['name']}: {status}")
        
        if has_dataset:
            available_trainings.append(script_info)
        else:
            missing_datasets.append(script_info)
    
    if not available_trainings:
        print("\n❌ 没有可用的数据集！请先下载数据集。")
        print("\n数据集下载地址:")
        print("  肺炎: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("  乳腺癌: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print("  脑肿瘤: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
        print("  疟疾: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
        return
    
    print(f"\n🚀 即将训练 {len(available_trainings)} 个模型")
    
    # 确认开始
    user_input = input("\n按 Enter 开始训练，输入 q 退出: ").strip().lower()
    if user_input == 'q':
        print("已取消")
        return
    
    # 开始训练
    total_start = time.time()
    results = {}
    
    for i, script_info in enumerate(available_trainings, 1):
        print(f"\n{'#'*60}")
        print(f"# 进度: {i}/{len(available_trainings)}")
        print(f"{'#'*60}")
        
        result = run_training(script_info)
        results[script_info['name']] = result
    
    total_time = (time.time() - total_start) / 60
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("📊 训练汇总")
    print("=" * 70)
    
    success_count = 0
    for name, result in results.items():
        if result == 'success':
            emoji = '✅'
            status = '成功'
            success_count += 1
        elif result == 'missing_dataset':
            emoji = '⏭️'
            status = '跳过(无数据集)'
        elif result == 'missing_script':
            emoji = '⏭️'
            status = '跳过(无脚本)'
        else:
            emoji = '❌'
            status = '失败'
        
        print(f"  {emoji} {name}: {status}")
    
    print(f"\n总用时: {total_time:.1f} 分钟")
    print(f"成功: {success_count}/{len(available_trainings)}")
    
    if success_count > 0:
        print("\n" + "=" * 70)
        print("🔄 下一步: 运行模型转换")
        print("=" * 70)
        print(f"\npython {MODELS_DIR}/convert_all.py")
        
        convert_input = input("\n是否现在运行转换? (y/n): ").strip().lower()
        if convert_input == 'y':
            subprocess.run([sys.executable, f'{MODELS_DIR}/convert_all.py'])

if __name__ == '__main__':
    main()
