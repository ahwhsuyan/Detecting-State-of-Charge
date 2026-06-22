import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
# from imblearn.over_sampling import ADASYN
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.optim import Adam
from options1 import get_options
from soc_dataset import SoCDataset
from reinforce_baseline import ExponentialBaseline
from attack_policy.DNNAgent import mal_rl_agent
from attack_policy.spoof_agent1 import mal_agent1
from attack_policy.spoof_agent2 import mal_agent2
from attack_policy.spoof_agent3 import mal_agent3
from attack_policy.spoof_agent4 import mal_agent4
from charging_env import charging_ev
from DetectionModelDNN import DetectionModelDNN
import os
from itertools import product
import json
import seaborn as sns
import matplotlib.patches as mpatches
import time
from thop import profile, clever_format


def train_dnn(opts):
  torch.random.manual_seed(opts.seed)
  np.random.seed(opts.seed)
  
  # 创建以当前时间命名的文件夹（精确到月份和分钟）
  current_time = time.strftime('%Y-%m-%d_%H-%M')
  time_based_dir = os.path.join('11.9begin', current_time)
  os.makedirs(time_based_dir, exist_ok=True)
  # 将时间文件夹路径保存到opts中，以便在其他函数中使用
  opts.time_based_dir = time_based_dir
  print(f"\n📁 本次运行结果将保存到: {time_based_dir}")
  
  # 显示当前使用的设备信息
  print(f"\n🔧 训练配置:")
  print(f"设备: {opts.device}")
  print(f"批次大小: {opts.batch_size}")
  print(f"训练轮数: {opts.n_epochs}")
  
  if opts.use_cuda:
    # 清理GPU缓存
    torch.cuda.empty_cache()
    print(f"GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
  print("-" * 50)
  if not os.path.exists(opts.save_dir) and not opts.eval_detect:
        os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
          json.dump(vars(opts), f, indent=True)

  train_dataset = torch.load(opts.train_dataset)
#  train_dataset = train_dataset.reshape(train_dataset.size(0) * train_dataset.size(1), -1)
  val_dataset = torch.load(opts.val_dataset)
  test_dataset = torch.load(opts.test_dataset)

#  val_dataset = val_dataset.reshape(val_dataset.size(0) * val_dataset.size(1), -1)
 

  train_loader = DataLoader(SoCDataset(train_dataset[:, :-1], train_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True,drop_last=True)
  val_loader = DataLoader(SoCDataset(val_dataset[:, :-1], val_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
  test_loader = DataLoader(SoCDataset(test_dataset[:, :-1], test_dataset[:, -1][:, None]), batch_size=opts.batch_size, shuffle=True)
  print(train_dataset.shape)

  # 统计模型参数量、FLOPs和MACs，并保存到para/model_profile.txt
  print("📦 正在创建模型...")
  model = DetectionModelDNN(opts.hidden_size, opts.num_timesteps, opts.p).to(opts.device)
  print("✅ 模型创建完成")
  
  print("📊 正在统计模型参数...")
  total_params = sum(p.numel() for p in model.parameters())
  print(f"✅ 模型参数量: {total_params:,}")
  
  input_tensor = torch.randn(1000, opts.num_timesteps).to(opts.device)  # 修正输入shape
  print("🔍 正在计算FLOPs和MACs...")
  try:
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs = flops // 2
    print(f"✅ FLOPs计算完成: {flops:,}, MACs: {macs:,}")
  except Exception as e:
    print(f"⚠️ FLOPs计算失败: {e}")
    flops, macs = 0, 0

  # 统计推理延迟和FPS（单样本检测延迟）
  print("⏱️ 正在测试推理延迟...")
  model.eval()
  num_runs = 100
  times = []
  with torch.no_grad():
    for i in range(num_runs):
      start = time.time()
      _ = model(input_tensor)
      end = time.time()
      times.append((end - start) * 1000)  # ms
      if (i + 1) % 20 == 0:
        print(f"  已完成 {i + 1}/{num_runs} 次测试...")
  avg_latency = sum(times) / len(times)
  std_latency = np.std(times)
  min_latency = min(times)
  max_latency = max(times)
  fps = 1000.0 / avg_latency if avg_latency > 0 else 0
  print(f"✅ 推理延迟测试完成:")
  print(f"   平均延迟: {avg_latency:.3f}ms ± {std_latency:.3f}ms")
  print(f"   最小延迟: {min_latency:.3f}ms, 最大延迟: {max_latency:.3f}ms")
  print(f"   FPS: {fps:.2f}")
  
  # 测试不同批次大小的检测延迟
  print("⏱️ 正在测试不同批次大小的检测延迟...")
  batch_sizes = [1, 10, 100, 1000]
  batch_latencies = {}
  for bs in batch_sizes:
    if bs <= input_tensor.size(0):
      test_input = input_tensor[:bs]
      times_batch = []
      with torch.no_grad():
        for _ in range(20):  # 每个批次大小测试20次
          start = time.time()
          _ = model(test_input)
          end = time.time()
          times_batch.append((end - start) * 1000)
      avg_batch_latency = sum(times_batch) / len(times_batch)
      per_sample_latency = avg_batch_latency / bs
      batch_latencies[bs] = {
        'total': avg_batch_latency,
        'per_sample': per_sample_latency
      }
      print(f"   批次大小 {bs}: 总延迟 {avg_batch_latency:.3f}ms, 单样本延迟 {per_sample_latency:.3f}ms")
  
  # 计算GPU内存使用
  print("💾 正在统计GPU内存使用...")
  if opts.use_cuda:
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
      _ = model(input_tensor)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    print(f"✅ GPU内存统计:")
    print(f"   峰值内存: {peak_memory:.2f} MB")
    print(f"   当前内存: {current_memory:.2f} MB")
  else:
    peak_memory = 0
    current_memory = 0

  # 打印计算开销信息
  print(f"\n💰 计算开销统计:")
  print(f"   参数量: {total_params:,} ({total_params/1e6:.2f}M)")
  print(f"   FLOPs: {flops:,} ({flops/1e9:.2f}G)")
  print(f"   MACs: {macs:,} ({macs/1e9:.2f}G)")
  if opts.use_cuda:
    print(f"   GPU峰值内存: {peak_memory:.2f} MB ({peak_memory/1024:.2f} GB)")
    print(f"   GPU当前内存: {current_memory:.2f} MB ({current_memory/1024:.2f} GB)")
  else:
    print(f"   GPU内存: 未使用GPU")
  
  # 将所有指标保存到统一的结果文件中
  results_file = os.path.join(time_based_dir, 'all_metrics.txt')
  print(f"\n💾 正在保存所有指标到 {results_file}...")
  
  with open(results_file, 'w', encoding='utf-8') as f:
    f.write(f"{'='*80}\n")
    f.write(f"模型训练结果汇总\n")
    f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*80}\n\n")
    
    # 1. 模型配置
    f.write(f"【模型配置】\n")
    f.write(f"{'-'*80}\n")
    f.write(f"模型名称: DetectionModelDNN\n")
    f.write(f"hidden_size: {opts.hidden_size}\n")
    f.write(f"input_size: {opts.num_timesteps}\n")
    f.write(f"dropout: {opts.p}\n")
    f.write(f"批次大小: {opts.batch_size}\n")
    f.write(f"训练轮数: {opts.n_epochs}\n")
    f.write(f"\n")
    
    # 2. 计算开销
    f.write(f"【计算开销】\n")
    f.write(f"{'-'*80}\n")
    f.write(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)\n")
    f.write(f"FLOPs: {flops:,} ({flops/1e9:.2f}G)\n")
    f.write(f"MACs: {macs:,} ({macs/1e9:.2f}G)\n")
    if opts.use_cuda:
      f.write(f"GPU峰值内存: {peak_memory:.2f} MB ({peak_memory/1024:.2f} GB)\n")
      f.write(f"GPU当前内存: {current_memory:.2f} MB ({current_memory/1024:.2f} GB)\n")
    else:
      f.write(f"GPU内存: 未使用GPU\n")
    f.write(f"\n")
    
    # 3. 推理延迟
    f.write(f"【推理延迟】\n")
    f.write(f"{'-'*80}\n")
    f.write(f"平均延迟: {avg_latency:.3f} ± {std_latency:.3f} ms\n")
    f.write(f"最小延迟: {min_latency:.3f} ms\n")
    f.write(f"最大延迟: {max_latency:.3f} ms\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write(f"批次延迟统计:\n")
    for bs, lat_info in batch_latencies.items():
      f.write(f"  批次大小 {bs}: 总延迟 {lat_info['total']:.3f}ms, 单样本延迟 {lat_info['per_sample']:.3f}ms\n")
    f.write(f"\n")
  
  print(f"✅ 初始指标已保存到 {results_file}")

  if opts.eval_only:
    model = DetectionModelDNN(opts.hidden_size, opts.num_timesteps, opts.p).to(opts.device)

    if opts.load_path is not None:
      load_data = torch.load(opts.load_path, map_location=torch.device(torch.device(opts.device)))
      model.load_state_dict(load_data)
    loss = nn.CrossEntropyLoss()
    val_acc, val_f1, val_loss, val_fpr, val_det_latency, val_per_sample_latency = eval(model, test_loader, loss, opts)
    print(f"测试集准确率: {val_acc:.4f}, F1分数: {val_f1:.4f}, 误报率: {val_fpr:.4f}")
  elif opts.train_plots:
    colors = sns.color_palette()
    models = ['Model 1', 'Model 2', 'Model 3']
    plt.figure(1)
    sns.set(style="darkgrid")
    plots = []
    legend_patches = []
    for i, model in enumerate(models):
      model_train_loss = np.load(f"train_loss_{model[-1]}.npy")
    # plt.title(f"Num Cars {opts.num_cars} arrival rate : {opts.lamb}")
      line = sns.tsplot(data=np.array(model_train_loss), color=colors[i])
      patch = mpatches.Patch(color=colors[i], label=model)
      legend_patches.append(patch)
    plt.legend(handles=legend_patches, title="Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.savefig(opts.save_dir + "/training_loss.pdf", dpi=400)

    plt.figure(2)

    legend_patches = []
    for i, model in enumerate(models):
      model_train_loss = np.load(f"val_acc_{model[-1]}.npy")
    # plt.title(f"Num Cars {opts.num_cars} arrival rate : {opts.lamb}")
      line = sns.tsplot(data=np.array(model_train_loss), color=colors[i])
      patch = mpatches.Patch(color=colors[i], label=model)
      legend_patches.append(patch)
    plt.legend(handles=legend_patches, title="Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Accuracy")

    plt.savefig(opts.save_dir + "/validation_accuracy.pdf", dpi=400)

  elif opts.tune:
    PARAM_GRID = list(product(
            [0.01, 0.001, 0.0001, 0.00001, 0.02, 0.002, 0.0002, 0.00002, 0.03, 0.003, 0.0003, 0.00003, 0.004, 0.0004, 0.00004],  # learning_rate
            [0., 0.5, 0.6, 0.7, 0.75, 0.8, 0.85],  # dropout rate
            [1.0, 0.99, 0.98, 0.97, 0.96, 0.95]  # lr decay
        ))
    # total number of slurm workers detected
    # defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # this worker's array index. Assumes slurm array job is zero-indexed
    # defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    SCOREFILE = os.path.expanduser(f"./val_acc_with_syn.csv")
    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):
      torch.manual_seed(opts.seed)
      np.random.seed(opts.seed)
      params = PARAM_GRID[param_ix]
      opts.p = params[1]
      opts.lr_model = params[0]
      opts.lr_decay = params[2]

      model, _, _, _, _, _, _ = train_epoch(train_loader, val_loader, opts)
      val_acc, val_f1, val_loss, val_fpr, val_det_latency, val_per_sample_latency = eval(model, val_loader, nn.CrossEntropyLoss(), opts)
      # if avg_r > max_val:
      #   best_params = params
      #   max_val = avg_r

      with open(SCOREFILE, "a") as f:
        f.write(f'{",".join(map(str, params + (val_acc,)))}\n')
  else:
    print("\n🚀 开始训练...")
    print("=" * 50)
    seeds = [200, 890, 786, 4872]
    best_val_acc = 0
    all_train_l, all_val_l, all_val_ac, all_train_ac = [], [], [], []
    for seed in seeds:
      print(f"\n🌱 使用随机种子: {seed}")
      curr_model, curr_train_l, curr_val_l, curr_val_ac, curr_train_ac, max_val_acc, max_val_f1 = train_epoch(train_loader, val_loader, opts, seed=seed)

      all_train_l.append(curr_train_l)
      all_val_l.append(curr_val_l)
      all_val_ac.append(curr_val_ac)
      all_train_ac.append(curr_train_ac)
      if max_val_acc > best_val_acc:
        best_model = curr_model
        best_val_acc = max_val_acc

        best_train_l, best_val_l, best_val_ac, best_train_ac = curr_train_l, curr_val_l, curr_val_ac, curr_train_ac
        torch.save(best_model.state_dict(), opts.save_dir + "/best_model_overall.pt")
        
        print("Cur Best Val Acc: ", best_val_acc)
        
        # 更新所有种子中最佳的结果到统一的结果文件
        time_based_dir = getattr(opts, 'time_based_dir', os.path.join('11.9begin', time.strftime('%Y-%m-%d_%H-%M')))
        results_file = os.path.join(time_based_dir, 'all_metrics.txt')
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n【所有种子中的最佳结果】\n")
            f.write(f"{'-'*80}\n")
            f.write(f"最佳种子: {seed}\n")
            f.write(f"Best Overall Validation Accuracy: {best_val_acc:.4f}\n")
            f.write(f"{'='*80}\n")
        
      # 将每次训练的损失和准确率写入文本文件
      with open(opts.save_dir + "/train_loss.txt", "a") as f:
          f.write(f"Seed {seed}: {curr_train_l}\n")

      with open(opts.save_dir + "/val_loss.txt", "a") as f:
          f.write(f"Seed {seed}: {curr_val_l}\n")

      with open(opts.save_dir + "/val_accuracy.txt", "a") as f:
          f.write(f"Seed {seed}: {curr_val_ac}\n")

      with open(opts.save_dir + "/train_accuracy.txt", "a") as f:
          f.write(f"Seed {seed}: {curr_train_ac}\n")

    sns.set_style("darkgrid")
    plt.figure(1)
    line1, *_ = plt.plot(np.arange(opts.n_epochs), np.array(best_train_l))
    line2, *_ = plt.plot(np.arange(opts.n_epochs), np.array(best_val_l))
    plt.title("Loss During Training")
    plt.xlabel("Batch Step")
    plt.ylabel("Loss")
    plt.legend((line1, line2), ("Training Loss", "Validation Loss"))
    plt.savefig(opts.save_dir + "/train_loss.pdf", dpi=1200)
    plt.figure(2)
    line1, *_ = plt.plot(np.arange(opts.n_epochs), best_train_ac)
    line2, *_ = plt.plot(np.arange(opts.n_epochs), best_val_ac)
    plt.title("Accuracy During Training")
    plt.xlabel("Batch Step")
    plt.ylabel("Accuracy")
    plt.legend((line1, line2), ("Training Accuracy", "Validation Accuracy"))
    plt.savefig(opts.save_dir + "/accuracy.pdf", dpi=1200)
    print(np.array(all_val_ac).shape)
    # 保存到原始目录
    np.save(opts.save_dir + "/val_acc.npy", np.array(all_val_ac))
    np.save(opts.save_dir + "/val_loss.npy", np.array(all_val_l))
    np.save(opts.save_dir + "/train_loss.npy", np.array(all_train_l))
    np.save(opts.save_dir + "/train_acc.npy", np.array(all_train_ac))
    
    # 保存训练指标数组到时间命名的文件夹
    time_based_dir = getattr(opts, 'time_based_dir', os.path.join('11.9begin', time.strftime('%Y-%m-%d_%H-%M')))
    os.makedirs(time_based_dir, exist_ok=True)
    np.save(os.path.join(time_based_dir, "val_acc.npy"), np.array(all_val_ac))
    np.save(os.path.join(time_based_dir, "val_loss.npy"), np.array(all_val_l))
    np.save(os.path.join(time_based_dir, "train_loss.npy"), np.array(all_train_l))
    np.save(os.path.join(time_based_dir, "train_acc.npy"), np.array(all_train_ac))
    
    # 更新统一的结果文件
    results_file = os.path.join(time_based_dir, 'all_metrics.txt')
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n【训练过程统计】\n")
        f.write(f"{'-'*80}\n")
        f.write(f"所有种子的平均最终验证准确率: {np.mean([ac[-1] for ac in all_val_ac]):.4f}\n")
        f.write(f"所有种子的平均最终验证损失: {np.mean([loss[-1] for loss in all_val_l]):.4f}\n")
        f.write(f"训练指标数组已保存: val_acc.npy, val_loss.npy, train_loss.npy, train_acc.npy\n")
        f.write(f"{'='*80}\n")
    print(f"✅ 所有训练指标已保存到 {time_based_dir} 文件夹")



def train_epoch(train_loader, val_loader, opts, seed=None):
  if seed is not None:
    torch.random.manual_seed(seed)
    np.random.seed(seed)
  print(f"📦 初始化模型 (hidden_size={opts.hidden_size}, input_size={opts.num_timesteps}, dropout={opts.p})...")
  model = DetectionModelDNN(opts.hidden_size, opts.num_timesteps, opts.p).to(opts.device)
  print("✅ 模型初始化完成")

  optimizer = Adam(model.parameters(), lr=opts.lr_model)
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: opts.lr_decay ** epoch
  )
  l = nn.CrossEntropyLoss()
  train_l = []
  val_l = []
  val_ac = []
  train_ac = []
  max_acc = 0.
  max_f1 = 0.
  val_acc = 0
  val_loss = 0
  # 用于跟踪最佳指标对应的其他指标
  best_fpr = 0.0
  best_det_latency = 0.0
  best_per_sample_latency = 0.0
  best_metrics = {}  # 存储最佳准确率对应的所有指标
  for epoch in range(opts.n_epochs):
    val_acc = 0.
    val_f1 = 0.
    train_acc = 0.
    total_train = 0
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{opts.n_epochs} [Train]', 
                     ncols=100, leave=False, disable=False):
      model.train()
      x = x.to(device=opts.device)
      y = y.to(device=opts.device)
      optimizer.zero_grad()
      output = model(x.float())
      train_loss = l(output, y.flatten().long())
      train_loss.backward()
      optimizer.step()
      
      # 计算训练准确率
      train_acc += torch.sum((output.argmax(1) == y.squeeze(1)).float()).item()
      total_train += x.size(0)
    
    # 计算平均训练准确率
    train_acc = train_acc / total_train
    
    if not opts.tune:
      val_acc, val_f1, val_loss, val_fpr, val_det_latency, val_per_sample_latency = eval(model, val_loader, l, opts)
      
      # 分别跟踪最高准确率和最高F1分数
      if val_acc > max_acc:
        max_acc = val_acc
        torch.save(model.state_dict(), opts.save_dir + "/best_model_acc.pt".format(epoch))
        print(f"当前最佳准确率（Best Accuracy）: {max_acc:.4f}")
        # 更新最佳准确率对应的其他指标
        best_fpr = val_fpr
        best_det_latency = val_det_latency
        best_per_sample_latency = val_per_sample_latency
        # 保存当前最佳指标对应的所有评估指标
        best_metrics = getattr(opts, 'current_metrics', {}).copy()
        
      if val_f1 > max_f1:
        max_f1 = val_f1
        torch.save(model.state_dict(), opts.save_dir + "/best_model_f1.pt".format(epoch))
        print(f"当前最佳F1分数（Best F1 Score）: {max_f1:.4f}")
    train_l.append(train_loss.detach().item())
    val_l.append(torch.tensor(val_loss).mean().item())
    val_ac.append(val_acc)
    train_ac.append(train_acc)
    print("\nEpoch {}: Train Loss : {} Train Acc : {} Val Accuracy : {} Val F1 : {} Val FPR : {}".format(
        epoch, train_loss, train_acc, val_acc, val_f1, val_fpr if not opts.tune else 0.0))
    lr_scheduler.step()
  
  # 训练完成后，保存本次运行的最佳结果到统一的结果文件
  if not opts.tune:
    # 获取时间命名的文件夹（从opts中获取）
    time_based_dir = getattr(opts, 'time_based_dir', None)
    if time_based_dir is None:
      # 如果不存在，创建新的时间文件夹
      current_time = time.strftime('%Y-%m-%d_%H-%M')
      time_based_dir = os.path.join('11.9begin', current_time)
      opts.time_based_dir = time_based_dir
    
    os.makedirs(time_based_dir, exist_ok=True)
    results_file = os.path.join(time_based_dir, 'all_metrics.txt')
    
    # 使用最佳准确率对应的指标（如果存在），否则使用最后一次的指标
    if best_metrics:
      final_metrics = best_metrics
    else:
      final_metrics = getattr(opts, 'current_metrics', {})
    
    # 追加本次运行的最佳结果到统一的结果文件
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n【本次运行最佳结果】\n")
        f.write(f"{'-'*80}\n")
        f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if seed is not None:
          f.write(f"随机种子: {seed}\n")
        f.write(f"最佳验证准确率 (Best Validation Accuracy): {max_acc:.4f}\n")
        f.write(f"最佳F1分数 (Best F1 Score): {max_f1:.4f}\n")
        f.write(f"精确率 (Precision): {final_metrics.get('precision', 0.0):.4f}\n")
        f.write(f"召回率 (Recall): {final_metrics.get('recall', 0.0):.4f}\n")
        f.write(f"误报率 (False Positive Rate): {best_fpr:.4f}\n")
        f.write(f"漏报率 (False Negative Rate): {final_metrics.get('fnr', 0.0):.4f}\n")
        f.write(f"检测延迟 (Detection Latency): {best_det_latency:.3f} ms/批次\n")
        f.write(f"单样本检测延迟 (Per-Sample Latency): {best_per_sample_latency:.3f} ms\n")
        if final_metrics.get('confusion_matrix'):
          cm = final_metrics['confusion_matrix']
          f.write(f"混淆矩阵 (Confusion Matrix):\n")
          f.write(f"  TN={cm[0][0]}, FP={cm[0][1]}\n")
          f.write(f"  FN={cm[1][0]}, TP={cm[1][1]}\n")
        f.write(f"{'='*80}\n")
    print(f"✅ 本次运行最佳结果已追加到 {results_file}")
  
  return model, train_l, val_l, val_ac, train_ac, max_acc, max_f1


def eval(model, val_loader, loss, opts):
  total = 0
  val_acc = 0
  val_loss = []
  all_preds = []
  all_labels = []
  
  # 用于统计检测延迟
  detection_times = []
  
  for x, y in tqdm(val_loader, desc='[Validation]', ncols=100, leave=False, disable=False):
    model.eval()
    x = x.to(device=opts.device)
    y = y.to(device=opts.device)
    
    # 测量检测延迟
    start_time = time.time()
    output = model(x.float()).detach()
    end_time = time.time()
    detection_times.append((end_time - start_time) * 1000)  # ms
    
    val_l = loss(output, y.flatten().long())
    val_acc += torch.sum((output.argmax(1) == y.squeeze(1)).float()).item()
    total += x.size(0)
    val_loss.append(val_l)
    
    # 收集预测和标签用于计算F1分数和混淆矩阵
    preds = torch.argmax(output, dim=1).cpu().numpy()
    labels = y.squeeze(1).cpu().numpy()
    all_preds.extend(preds)
    all_labels.extend(labels)
  
  print(f"实际验证样本数: {total}, 配置的val_size: {opts.val_size}")
  
  # 修复：应该除以实际处理的样本数total，而不是固定的opts.val_size
  val_acc = val_acc / total if total > 0 else 0.0
  
  # 计算F1分数和其他指标
  from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
  
  # 对于二分类问题，使用'macro'或'binary'都可以，但'macro'更通用
  # 如果数据不平衡，'macro'会更好；如果平衡，两者结果接近
  val_f1 = f1_score(all_labels, all_preds, average='macro')
  
  # 验证：使用sklearn的accuracy_score来确认计算是否正确
  sklearn_acc = accuracy_score(all_labels, all_preds)
  
  # 如果val_acc和sklearn_acc差异较大，说明计算有问题
  if abs(val_acc - sklearn_acc) > 1e-5:
    print(f"⚠️ 警告: 计算的准确率({val_acc:.6f})与sklearn准确率({sklearn_acc:.6f})不一致！")
    val_acc = sklearn_acc  # 使用sklearn的结果作为准确值
  
  # 计算混淆矩阵
  cm = confusion_matrix(all_labels, all_preds)
  
  # 计算误报率（False Positive Rate, FPR）
  # FPR = FP / (FP + TN)，其中FP是假阳性，TN是真阴性
  if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 真阳性率（召回率）
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # 假阴性率（漏报率）
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
  else:
    # 如果只有一类，设置默认值
    FPR = 0.0
    TPR = 0.0
    FNR = 0.0
    precision = 0.0
    recall = 0.0
  
  # 计算检测延迟统计
  avg_detection_latency = np.mean(detection_times) if detection_times else 0.0
  std_detection_latency = np.std(detection_times) if detection_times else 0.0
  # 计算单样本延迟：总延迟 / 批次数量，然后除以平均批次大小
  avg_batch_size = total / len(detection_times) if detection_times and len(detection_times) > 0 else 1.0
  per_sample_latency = avg_detection_latency / avg_batch_size if avg_batch_size > 0 else 0.0
  
  # 保存当前评估的指标到opts中
  if not hasattr(opts, 'current_metrics'):
    opts.current_metrics = {}
  opts.current_metrics = {
    'accuracy': val_acc,
    'f1': val_f1,
    'precision': precision,
    'recall': recall,
    'fpr': FPR,
    'fnr': FNR,
    'detection_latency': avg_detection_latency,
    'per_sample_latency': per_sample_latency,
    'confusion_matrix': cm.tolist() if cm.shape == (2, 2) else None
  }
  
  # 打印详细指标
  print(f"\n📊 详细评估指标:")
  print(f"   准确率 (Accuracy): {val_acc:.4f}")
  print(f"   F1分数 (F1 Score): {val_f1:.4f}")
  print(f"   精确率 (Precision): {precision:.4f}")
  print(f"   召回率 (Recall/TPR): {recall:.4f}")
  print(f"   误报率 (False Positive Rate): {FPR:.4f}")
  print(f"   漏报率 (False Negative Rate): {FNR:.4f}")
  print(f"   检测延迟: {avg_detection_latency:.3f} ± {std_detection_latency:.3f} ms/批次")
  print(f"   单样本检测延迟: {per_sample_latency:.3f} ms")
  if cm.shape == (2, 2):
    print(f"   混淆矩阵:")
    print(f"      TN={TN}, FP={FP}")
    print(f"      FN={FN}, TP={TP}")
  
  # 不在这里保存评估指标，只在训练完成后保存最佳结果
  
  return val_acc, val_f1, val_loss, FPR, avg_detection_latency, per_sample_latency



if __name__ == "__main__":
    train_dnn(get_options())
