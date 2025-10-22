import torch
import torch.nn.functional as F
import numpy as np
import os
import json

class UDFGrid:
    def __init__(self, scene_path, device="cuda"):
        self.scene_path = scene_path
        self.device = device
        self.udf_grid = None
        self.min_bound = None
        self.max_bound = None
        self.resolution = None
        
        self._load_grid()

    def _load_grid(self):
        """加载 UDF 网格文件和元数据。"""
        grid_path = os.path.join(self.scene_path, "udf_grid.npy")
        info_path = os.path.join(self.scene_path, "udf_grid_info.json")

        if not os.path.exists(grid_path) or not os.path.exists(info_path):
            print(f"[UDF] ⚠️ 警告: 场景 '{os.path.basename(self.scene_path)}' 未找到 UDF 文件。约束将禁用。")
            return

        print(f"[UDF] ✅ 正在加载 UDF 网格: {grid_path}")
        
        # 1. 加载元数据
        with open(info_path, 'r') as f:
            info = json.load(f)
            self.min_bound = torch.tensor(info['min_bound'], dtype=torch.float32, device=self.device)
            self.max_bound = torch.tensor(info['max_bound'], dtype=torch.float32, device=self.device)
            self.resolution = info['resolution']

        # 2. 加载 NPY 数据，并转移到 GPU
        udf_np = np.load(grid_path)
        # 将 [D, H, W] 的 NumPy 数组转换为 PyTorch [1, 1, D, H, W] 或 [1, D, H, W]
        # 为了方便 F.grid_sample (它需要 N, C, D, H, W)，我们使用 [1, 1, D, H, W]
        self.udf_grid = torch.from_numpy(udf_np).unsqueeze(0).unsqueeze(0).to(self.device).float()

    # def query(self, xyz_world: torch.Tensor) -> torch.Tensor:
    #     """
    #     核心函数：查询世界坐标系下的点到稠密点云的 UDF 距离。
    #     使用三线性插值，返回一个可微分的距离张量。
        
    #     参数:
    #     - xyz_world: (N, 3) 张量，世界坐标系下的高斯中心坐标。
        
    #     返回:
    #     - (N, 1) 张量，每个点对应的 UDF 距离。
    #     """
    #     if self.udf_grid is None:
    #         # 如果UDF未加载，返回一个零张量（或inf张量）以禁用约束
    #         return torch.zeros_like(xyz_world[:, :1]) 

    #     # --- [ 步骤 1: 将世界坐标 XYZ 映射到归一化网格坐标 [-1, 1] ] ---
        
    #     # 归一化公式: 2 * (X_world - X_min) / (X_max - X_min) - 1
    #     # 这就是将 [X_min, X_max] 线性映射到 [-1, 1]
        
    #     # 1. 计算范围 (Span)
    #     span = self.max_bound - self.min_bound
        
    #     # 2. 映射
    #     # x_norm = 2 * (x_world - x_min) / span - 1
    #     xyz_norm = 2 * (xyz_world - self.min_bound) / span - 1
        
    #     # --- [ 步骤 2: 准备 F.grid_sample 的输入 ] ---
        
    #     # grid_sample 要求坐标是 (N, D, H, W, 3) 形状，且维度顺序是 (Z, Y, X)
    #     # 我们需要将 (N, 3) 形状的 (X, Y, Z)_norm 转换为 (1, N, 1, 1, 3) 形状的 (X_norm, Y_norm, Z_norm)
        
    #     # 3. 维度转换: [N, 3] -> [N, 1, 1, 1, 3]
    #     # grid_sample 的坐标输入形状必须是 (N_batch, D, H, W, 3)
    #     # 我们的目标是查询 N 个点，所以 N_batch = N, D=H=W=1
        
    #     # 4. 调整坐标顺序: PyTorch 的 grid_sample 期望 (X, Y, Z) 顺序，对应网格的 (W, H, D)
    #     # 我们这里的 UDF 网格是 (D, H, W) 顺序
    #     # 实际 grid_sample 内部的维度顺序是 (W, H, D)
    #     # 为了让 PyTorch 的 grid_sample 理解我们的 (D, H, W) 形状，我们必须将查询坐标的顺序设置为 (W, H, D) 对应的归一化坐标。
    #     # 也就是 [Z_norm, Y_norm, X_norm]
        
    #     # 我们需要 PyTorch 的 (x, y, z) 坐标，其中 x 对应最里层维度 (W), y 对应中间维度 (H), z 对应最外层维度 (D)
        
    #     # 我们的网格形状是 (D, H, W) 
    #     # grid_sample 期望的 coordinate order 是 (W, H, D) (即 X, Y, Z)
    #     # 所以我们传入的坐标顺序应该是 [X_norm, Y_norm, Z_norm]
        
    #     # 5. 转换并插值
    #     # [N, 3] -> [N, 1, 1, 1, 3]
    #     query_coords = xyz_norm.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
    #     # PyTorch 的 grid_sample 期望坐标顺序是 (W, H, D) 对应 (X, Y, Z)
    #     # UDF grid 形状是 (1, 1, D, H, W). grid_sample 期望的 grid 形状是 (N, C, D, H, W)
        
    #     # 实际需要的 coordinate order: (X, Y, Z)
    #     # 归一化坐标: [X_norm, Y_norm, Z_norm]
    #     # 我们需要将 [Z_norm, Y_norm, X_norm] 传入 grid_sample (Z, Y, X 顺序)
        
    #     # 最终的坐标顺序应该是 [Z_norm, Y_norm, X_norm]
    #     # 但是 grid_sample 的坐标顺序总是 (x, y, z) 对应 (W, H, D)。 
    #     # 为了匹配我们的 (D, H, W) 形状，我们传入 [Z_norm, Y_norm, X_norm]
    #     query_coords = xyz_norm[:, [2, 1, 0]].unsqueeze(0).unsqueeze(0).unsqueeze(0) # 形状: [1, 1, 1, N, 3] 
    #     # 不行，这样形状不对。

    #     # 正确方法：使用 N 个点作为 Batch Size
    #     # UDF Grid shape: [1, 1, D, H, W]
    #     # Input Coords shape: [N, 1, 1, 3] (X, Y, Z)
        
    #     # [N, 3] -> [N, 1, 1, 1, 3]
    #     # 调整顺序: [X, Y, Z] -> [W, H, D]
    #     # 我们需要传入 [W_norm, H_norm, D_norm] 
    #     # 对应 [X_norm, Y_norm, Z_norm]
    #     query_coords_reordered = xyz_norm[:, [2, 1, 0]].unsqueeze(1).unsqueeze(1) # [N, 1, 1, 3]
        
    #     # 5D 张量: (N_batch, C_in, D_in, H_in, W_in)
    #     # 我们需要一个 [N, 1, 1, 1, 3] 的坐标张量
        
    #     # 终极正确方法：重塑张量并使用 grid_sample
    #     # UDF Grid shape: [1, 1, D, H, W]
    #     # Query Coords shape: [N, 3] (X, Y, Z)
        
    #     # Query: [1, N, 1, 1, 3] - N queries, 1 batch, 1D, 1H
        
    #     # 重塑并确保顺序正确 (Z, Y, X 对应 D, H, W)
    #     query_coords_reordered = xyz_norm[:, [2, 1, 0]].unsqueeze(0).unsqueeze(0).unsqueeze(0).transpose(0, 3).squeeze(0) # [1, N, 1, 3] -> 无法做到
        
    #     # 简单的方法：使用 [1, N, 1, 1, 3] 的形状进行三线性插值
    #     # 调整顺序: [X, Y, Z] -> [W, H, D] -> [Z, Y, X] (对应 (D, H, W))
    #     coords_for_sample = xyz_norm[:, [2, 1, 0]].unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, N, 3]
        
    #     # 6. 执行插值
    #     # grid_sample 返回 [1, 1, 1, N]
    #     udf_values = F.grid_sample(self.udf_grid, coords_for_sample, mode='bilinear', padding_mode='border', align_corners=True).squeeze() # [N]
        
    #     return udf_values.unsqueeze(1) # [N, 1]
    def query(self, xyz_world: torch.Tensor) -> torch.Tensor:
    """
    核心函数：查询世界坐标系下的点到稠密点云的 UDF 距离。
    使用三线性插值，返回一个可微分的距离张量。
    
    参数:
    - xyz_world: (N, 3) 张量，世界坐标系下的高斯中心坐标。
    
    返回:
    - (N, 1) 张量，每个点对应的 UDF 距离。
    """
    if self.udf_grid is None:
        # 如果UDF未加载，返回一个零张量以禁用约束
        return torch.zeros(xyz_world.shape[0], 1, device=xyz_world.device)

    # --- [ 步骤 1: 将世界坐标 XYZ 映射到归一化网格坐标 [-1, 1] ] ---
    span = self.max_bound - self.min_bound
    xyz_norm = 2.0 * (xyz_world - self.min_bound) / span - 1.0
    
    # --- [ 步骤 2: 准备 F.grid_sample 的输入 ] ---
    # F.grid_sample 需要一个形状为 (N, D_out, H_out, W_out, 3) 的坐标网格。
    # 我们要查询 N 个独立的点，可以看作是 N 个 1x1x1 的输出网格。
    # 因此，我们将 (N, 3) 的坐标变形为 (N, 1, 1, 1, 3)。
    # 坐标顺序 (x, y, z) 天然对应 grid 的 (W, H, D) 维度，无需调换。
    query_coords = xyz_norm.unsqueeze(1).unsqueeze(1).unsqueeze(1) # Shape: [N, 1, 1, 1, 3]

    # --- [ 步骤 3: 执行插值 ] ---
    # UDF Grid Shape: [1, 1, D, H, W]
    # Query Coords Shape: [N, 1, 1, 1, 3]
    # F.grid_sample 将返回形状为 [N, 1, 1, 1, 1] 的张量
    udf_values = F.grid_sample(
        self.udf_grid, 
        query_coords, 
        mode='bilinear', # 'bilinear' 在 3D 模式下等同于 'trilinear'
        padding_mode='border', # 'border' 会将越界坐标钳位到 [-1, 1] 边界，是安全的做法
        align_corners=False    # 必须为 False 来匹配我们的归一化公式
    )
    
    # 将 [N, 1, 1, 1, 1] 挤压为 [N, 1]
    return udf_values.squeeze(-1).squeeze(-1).squeeze(-1) # Shape: [N, 1]