import matplotlib.pyplot as plt


# 初始化绘图
def init_plot(lr):
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    x_list = [] # x轴数据
    y_list = [] # y轴数据
    line, = ax.plot(x_list, y_list, 'b-', linewidth=2, label='Perplexity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'bahdanau attention Training loss vs Epoch (lr={lr})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax, line, x_list, y_list
    # fig是画布
    # ax是坐标系区域(axis是带刻度的图表框)，一个fig 上可以有多个 ax
    # line是线对象
    # x_list和y_list是数据列表


# 更新绘图
def update_plot(x_item, y_item, x_list, y_list, line, ax):
    x_list.append(x_item)
    y_list.append(y_item)
    line.set_xdata(x_list)
    line.set_ydata(y_list)
    ax.set_xlim(0, x_item + 2)  # 确保x轴范围包含当前epoch，右边预留2个单位
    ax.set_ylim(0, max(y_list) * 1.1 if y_list else 1)  # 防止空列表报错，y轴预留10%
    plt.draw()
    plt.pause(0.01)

# 关闭绘图
def close_plot():
    plt.ioff()  # 关闭交互模式->恢复默认行为
    plt.show()  # 阻塞，保持窗口打开直到用户手动关闭