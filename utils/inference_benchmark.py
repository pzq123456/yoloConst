from ultralytics import YOLO
import os
import time
import numpy as np

def inference_with_model(model_path, image_paths, model_name="Model", half=False):
    """
    使用指定模型进行推理并测量速度
    
    Args:
        model_path: 模型路径
        image_paths: 图片路径列表
        model_name: 模型名称（用于显示）
        half: 是否使用半精度
    
    Returns:
        dict: 包含推理时间和结果统计
    """
    print(f"\n{'='*60}")
    print(f"加载模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"{'='*60}")
    
    # 加载模型
    start_load = time.time()
    model = YOLO(model_path, task="detect")
    load_time = time.time() - start_load
    print(f"模型加载耗时: {load_time:.3f} 秒")
    
    # 推理统计
    inference_times = []
    total_objects = 0
    successful_images = 0
    
    # 逐张推理
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"⚠️ 图片不存在: {img_path}")
            continue
            
        # 单张推理并计时
        start_time = time.time()
        results = model.predict(
            source=img_path,
            save=True,
            conf=0.3,
            half=half,
            verbose=False,  # 减少输出
        )
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # 统计结果
        num_objects = len(results[0].boxes) if results[0].boxes else 0
        total_objects += num_objects
        successful_images += 1
        
        print(f" 图片 {i+1}: {os.path.basename(img_path)} - "
              f"耗时 {inference_time:.3f}s, 检测到 {num_objects} 个目标")
    
    # 计算统计数据
    if inference_times:
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
    else:
        avg_time = std_time = min_time = max_time = fps = 0
    
    return {
        'model_name': model_name,
        'load_time': load_time,
        'inference_times': inference_times,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'total_objects': total_objects,
        'successful_images': successful_images,
        'total_images': len(image_paths)
    }

def compare_models(image_paths):
    """
    对比 PyTorch 和 TensorRT 模型的推理速度
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径
    pt_model_path = os.path.join(current_dir, "../model/best.pt")
    engine_model_path = os.path.join(current_dir, "../model/best.engine")
    
    # 检查模型是否存在
    models_to_test = []
    if os.path.exists(pt_model_path):
        models_to_test.append((pt_model_path, "PyTorch (FP32)", False))
    else:
        print(f"⚠️ PyTorch 模型不存在: {pt_model_path}")
    
    if os.path.exists(engine_model_path):
        models_to_test.append((engine_model_path, "TensorRT (FP16)", True))
    else:
        print(f"⚠️ TensorRT 模型不存在: {engine_model_path}")
    
    if not models_to_test:
        print("❌ 没有找到任何可用的模型文件！")
        return
    
    # 检查图片是否存在
    valid_images = [img for img in image_paths if os.path.exists(img)]
    if not valid_images:
        print("❌ 没有找到任何有效的图片文件！")
        return
    
    print(f"\n📸 将使用 {len(valid_images)} 张图片进行测试:")
    for img in valid_images:
        print(f"  - {os.path.basename(img)}")
    
    # 测试所有模型
    results = {}
    for model_path, model_name, half in models_to_test:
        results[model_name] = inference_with_model(
            model_path, 
            valid_images, 
            model_name, 
            half
        )
    
    # 打印对比结果
    print(f"\n{'='*60}")
    print("📊 推理速度对比结果")
    print(f"{'='*60}")
    print(f"{'模型类型':<20} {'平均耗时(s)':<12} {'FPS':<10} {'最快(s)':<10} {'最慢(s)':<10} {'标准差':<10}")
    print("-" * 80)
    
    for model_name, stats in results.items():
        print(f"{model_name:<20} {stats['avg_time']:<12.3f} "
              f"{stats['fps']:<10.1f} {stats['min_time']:<10.3f} "
              f"{stats['max_time']:<10.3f} {stats['std_time']:<10.3f}")
    
    print("-" * 80)
    
    # 速度提升对比
    if len(results) == 2:
        pt_stats = results.get("PyTorch (FP32)")
        trt_stats = results.get("TensorRT (FP16)")
        
        if pt_stats and trt_stats and pt_stats['avg_time'] > 0:
            speedup = pt_stats['avg_time'] / trt_stats['avg_time']
            print(f"\n🚀 TensorRT 相比 PyTorch 速度提升: {speedup:.2f}x")
            print(f"   (PyTorch: {pt_stats['avg_time']:.3f}s, TensorRT: {trt_stats['avg_time']:.3f}s)")
            
            # 详细统计
            print(f"\n📈 详细统计:")
            print(f"   TensorRT 加载速度: {trt_stats['load_time']:.3f}s (vs PyTorch: {pt_stats['load_time']:.3f}s)")
            print(f"   TensorRT 推理总目标数: {trt_stats['total_objects']} (vs PyTorch: {pt_stats['total_objects']})")
    
    # 保存结果到文件
    save_results_to_file(results, valid_images)

def save_results_to_file(results, image_paths):
    """保存测试结果到文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_file = os.path.join(current_dir, "benchmark_results.txt")
    
    with open(result_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("模型推理速度测试结果\n")
        f.write("="*80 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试图片数: {len(image_paths)}\n\n")
        
        for model_name, stats in results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  加载时间: {stats['load_time']:.3f}s\n")
            f.write(f"  平均推理时间: {stats['avg_time']:.3f}s\n")
            f.write(f"  FPS: {stats['fps']:.1f}\n")
            f.write(f"  最快: {stats['min_time']:.3f}s\n")
            f.write(f"  最慢: {stats['max_time']:.3f}s\n")
            f.write(f"  标准差: {stats['std_time']:.3f}s\n")
            f.write(f"  检测总目标: {stats['total_objects']}\n")
        
        # 速度提升
        if len(results) == 2:
            pt_stats = results.get("PyTorch (FP32)")
            trt_stats = results.get("TensorRT (FP16)")
            if pt_stats and trt_stats and pt_stats['avg_time'] > 0:
                speedup = pt_stats['avg_time'] / trt_stats['avg_time']
                f.write(f"\n🚀 速度提升: {speedup:.2f}x\n")
    
    print(f"\n💾 结果已保存到: {result_file}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 需要推理的图片路径列表
    image_paths = [
        os.path.join(current_dir, "../test/image.png"),
        os.path.join(current_dir, "../test/image copy.png"),
        os.path.join(current_dir, "../test/image copy 2.png"),
        os.path.join(current_dir, "../test/image copy 3.png"),
        os.path.join(current_dir, "../test/image1.png"),
    ]
    
    # 运行对比测试
    compare_models(image_paths)

if __name__ == "__main__":
    main()