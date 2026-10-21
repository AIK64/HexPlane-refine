# 深度监督接口

默认仍为 RGB 训练：`model.depth_loss: false`，深度目录为 `null`。
本接口读取已有深度文件，不会自动从 RGB 生成深度或对相对深度进行尺度对齐。

## DNeRF 文件约定

例如数据根目录是 `../dataset/DNeRF/lego`，RGB 为 `train/r_0.png`：

```text
lego/
  train/r_0.png
  test/r_0.png
  depth/train/r_0.npy
  depth/test/r_0.npy
```

每张深度必须是与原始 RGB 尺寸相同的二维单通道数组。
支持 `.npy`、`.pfm`、单通道 `.png`、`.tif`、`.tiff`，不支持伪彩色图。
目录的相对路径以数据集根目录为基准，绝对路径也可以。

```yaml
data:
  depth_train_dir: depth/train
  depth_test_dir: depth/test  # 没有测试深度时设为 null，仍可渲染预测深度
  depth_file_pattern: "{stem}.npy"
  depth_type: ray_distance
  depth_scale: 1.0

model:
  depth_loss: true
  depth_loss_weight: 1.0
  depth_render_mode: far

save_depth_raw: true
```

该片段可合并到原来的场景 YAML；文件准备好后再启用。
`depth_file_pattern` 支持 `{stem}`（如 `r_0`）、`{split}`（如 `train`）、
`{frame}`（如 `train/r_0`）。训练和测试可以分别提供或关闭深度。

## 深度数值约定

- `ray_distance`：相机到表面沿单位射线的距离，单位与相机位姿/场景坐标一致。
- `camera_z`：正的相机光轴深度。加载器用 `distance = camera_z / abs(unit_ray_z)` 转成沿射线距离。
- `depth_scale`：存储值乘以该系数得到场景单位。例如只有场景坐标以米表示时，毫米深度才应使用 `0.001`。
- 读取后用最近邻缩放到训练分辨率，避免混合边缘和无效深度。
- 非正数、NaN、Inf、完全透明的 RGB 背景，以及 near/far 采样范围外的参考深度均置零并忽略。
- 数据集中完全没有有效深度时直接报错；个别批次全无效时深度损失为零。
- 相对深度、逆深度不能直接当作以上两种深度；需要先转换并对齐尺度。

目前只接通 DNeRF 文件读取。`neural3D_NDC` 的度量深度到 NDC 射线参数转换尚未实现，
因此明确拒绝直接开启深度监督，而不是对不一致的量计算 MSE。
Trainer 的 `hierach` 分支支持从自定义数据集的
`all_depths[camera, time, pixel, 1]` 同步取值，但这不等于已接通 NDC 深度读取。

## 渲染、训练与评估

预测深度默认是 `sum(weight * z_vals) + (1 - sum(weight)) * far`。
旧代码使用方向的 `dz` 补背景，此处已修正；RGB 的合成方式没有改变。
可选 `conditional` 模式使用前景权重归一化深度；空射线返回 far。
训练和评估应使用相同的模式。读取旧 checkpoint 时会同步当前深度配置。

深度损失只对有限且为正的参考像素计算 MSE，加入 `depth_loss_weight` 后反向传播。
它是有参考深度的监督，不是相邻像素深度平滑损失。

评估目录新增：

- `depth_raw/*_depth.npy`：预测的 float32 原始深度，未经过伪彩色变换。
- `depth_raw/*_gt.npy`：如有参考深度，保存转换后的参考值；零代表无效。
- `*depth_metrics.json`：有效像素数、MAE、RMSE，以及各帧结果。
  汇总指标按所有有效像素计算，不对帧均值做简单平均；无有效像素时指标为 null。

相机路径渲染也保存原始预测深度。`save_depth_raw: false` 可关闭原始数组保存。
深度伪彩色图和视频仍保留，但不应用于定量计算。

## DA3 相对深度：images 模式

现有 images 采样器每次选一张图，再从该图选择 batch_size 条射线，
因此无需另建采样器或在 rays 模式里增加帧编号分组。

示例配置：`config/dnerf_slim_lego_relative_depth.yaml`。

```powershell
# 相同 images 采样的 RGB 对照实验
python main.py config=./config/dnerf_slim_lego_relative_depth.yaml model.depth_loss=False expname=lego_images_rgb

# 相对深度实验，前 3000 次迭代只用 RGB 和原有正则
python main.py config=./config/dnerf_slim_lego_relative_depth.yaml
```

关键组合是 `data.datasampler_type: images`、`data.depth_type: relative_camera_z`、
`model.depth_loss_type: relative`。其他采样模式暂不允许开启这个相对损失。
`depth_loss_start` 默认 0；示例设为 3000，`depth_loss_weight: 0.01` 是实验起始值。

相对相机 Z 深度不转换成射线距离，也不按 near/far 范围过滤。
训练时将预测射线深度乘以相机空间单位方向的 Z 分量绝对值，转换成相机 Z 深度。
在同一张图的有效采样像素上，将预测和参考各自减去中位数、除以平均绝对偏差，
再计算归一化 L1。有效点不足 8 个或参考深度为常量时跳过该批次的深度项。
这个损失约束相对形状，不提供真实距离尺度，也不自动消除伪深度中的几何错误。

相对模式的测试 JSON 报告逐帧归一化 L1，不报告距离单位下的 MAE/RMSE。
保存的预测原始数组仍是射线距离，参考数组则是相对相机 Z 深度，不能直接相减。
