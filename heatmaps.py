import torch
import numpy as np
import matplotlib.pyplot as plt

# matrices的shape必须是(num_rows, num_cols, H, W)
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5), cmap='Reds'):
    # 确保输入的 matrices 是一个 4 维张量
    if matrices.dim() != 4:
        raise ValueError("Input matrices must be a 4D tensor.")
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    # 确保 titles 是一个二维数组，与 matrices 形状匹配
    if titles is not None and titles.shape != (num_rows, num_cols):
        raise ValueError(f"Titles must be a 2D array with shape ({num_rows}, {num_cols}).")
    
    
    fig, axes = plt.subplots(           # 构造子图网格
        num_rows,                       # 子图网格的行数
        num_cols,                       # 子图网格的列数
        figsize=figsize,                # 图形的尺寸（宽，高），单位为英寸
        sharex=True,                    # 所有子图共享 x 轴刻度
        sharey=True,                    # 所有子图共享 y 轴刻度
        squeeze=False,                  # 即使子图只有一个，也返回二维数组形式的 axes
    )


    # axes[0]（第0行的3个子图） ↔ matrices[0]（第0行的3个5×5矩阵）
    # axes[1]（第1行的3个子图） ↔ matrices[1]（第1行的3个5×5矩阵）
    # [
    #     [ax00, ax01, ax02],   ←  matrices[0,0],matrices[0,1],matrices[0,2]
    #     [ax10, ax11, ax12]    ←  matrices[1,0],matrices[1,1],matrices[1,2]
    # ]
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)): 
        for j, (col_axes, col_matrix) in enumerate(zip(row_axes, row_matrices)): 
            # 在当前子图上绘制热图
            # col_matrix.detach().numpy()：将 PyTorch 张量从计算图中分离并转换为 NumPy 数组
            # cmap=cmap：使用指定的颜色映射来绘制热图
            # vmin=0, vmax=1：设置颜色映射的最小值和最大值为 0 和 1，确保所有值都映射到该范围内
            # 返回的 pcm 是一个 QuadMesh 对象，用于后续的颜色条添加
            # pcm = col_axes.imshow(col_matrix.detach().numpy(), cmap=cmap, vmin=0, vmax=1)
            pcm = col_axes.imshow(col_matrix.detach().numpy(), cmap=cmap)
            # 只在最后一行设置 x 轴标签
            if i == num_rows - 1:
                col_axes.set_xlabel(xlabel)
            # 只在第一列设置 y 轴标签
            if j == 0:
                col_axes.set_ylabel(ylabel) 
            # 如果提供了标题，设置子图标题
            if titles is not None:
                col_axes.set_title(titles[i,j])
    
    # 为所有子图添加一个共享的颜色条
    # pcm: 最后一个子图上调用 imshow 返回的 QuadMesh 对象，用于获取颜色映射信息
    # ax=axes: 指定颜色条关联的子图集合，即所有子图共享该颜色条
    # shrink=0.6: 将颜色条的长度缩小为原始长度的 60%，以适配整体布局
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    # 显示图形
    plt.show(block=True)


def main():     
    # ========== 构造 2x3 个 5x5 的热图数据 ==========
    H, W = 5, 5
    matrices = torch.zeros(2, 3, H, W)


    # 第0行
    matrices[0, 0] = torch.eye(H)                    # 对角线
    matrices[0, 1] = torch.ones(H, W) * 0.5          # 均匀浅色
    matrices[0, 2] = torch.arange(H).view(-1, 1).repeat(1, W) / H  # 行递增（列复制）

    # 第1行
    matrices[1, 0] = torch.arange(W).repeat(H, 1) / W  # 列递增（行复制）
    matrices[1, 1] = torch.randn(H, W).abs()           # 随机噪声（取绝对值好看些）
    matrices[1, 2] = torch.triu(torch.ones(H, W))      # 上三角

    # ========== 设置标题 ==========

    titles = np.array(["Diagonal", "Uniform", "Row Gradient","Col Gradient", "Noise", "Upper Tri"])
    titles = titles.reshape(2, 3)


    # ========== 调用函数 ==========
    show_heatmaps(
        matrices=matrices,
        xlabel='Keys',
        ylabel='Queries',
        titles=titles,
        figsize=(9, 6),   # 宽一些，适合 2x3 布局
        cmap='Reds'
    )


if __name__ == '__main__':
    main()
