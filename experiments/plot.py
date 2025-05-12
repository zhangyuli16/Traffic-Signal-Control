import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_color_codes()
import pandas as pd
import numpy as np
import os
import xml.etree.cElementTree as ET
from matplotlib.ticker import FuncFormatter

# 修改为你的数据路径
base_dir = r'F:\PHD\phd3\deeprl_signal_control\experiments\ma2c\data'
plot_dir = os.path.join(base_dir, 'plots')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
color_cycle = sns.color_palette()
COLORS = {'ma2c': color_cycle[0], 'ia2c': color_cycle[1], 'iqll': color_cycle[2],
          'iqld': color_cycle[3], 'greedy': color_cycle[4]}
TRAIN_STEP = 1e6

# Define window size for smoothing
window = 100


def plot_train_curve():
    """
    绘制MA2C的训练曲线
    """
    # 直接读取train_reward.csv文件
    file_path = os.path.join(base_dir, 'train_reward.csv')

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"正在读取文件: {file_path}")
    # 读取数据
    df = pd.read_csv(file_path)

    # 如果数据中包含多个算法，只筛选MA2C的数据
    if 'agent' in df.columns:
        df = df[df.agent == 'ma2c']
        print("已筛选MA2C数据")

    # 如果数据中有test_id列，只使用训练数据(test_id == -1)
    if 'test_id' in df.columns:
        df = df[df.test_id == -1]
        print("已筛选训练数据(test_id == -1)")

    # 确认有数据可用
    if len(df) == 0:
        print("没有找到有效的MA2C训练数据")
        return

    print(f"读取到 {len(df)} 行数据")

    # 绘图
    plt.figure(figsize=(9, 6))

    # 使用100作为滑动窗口大小计算平均值和标准差
    window_size = 100
    x_mean = df.avg_reward.rolling(window=window_size, min_periods=1).mean().values
    x_std = df.avg_reward.rolling(window=window_size, min_periods=1).std().values

    # 绘制主曲线
    plt.plot(df.step.values, x_mean, color=COLORS['ma2c'], linewidth=3, label='MA2C')

    # 添加标准差区域
    plt.fill_between(df.step.values, x_mean - x_std, x_mean + x_std,
                     facecolor=COLORS['ma2c'], edgecolor='none', alpha=0.1)

    # 自动确定Y轴范围
    ymin = np.nanmin(x_mean - 0.5 * x_std)
    ymax = np.nanmax(x_mean + 0.5 * x_std)

    # 设置图表范围
    plt.xlim([0, df.step.max()])
    plt.ylim([ymin, ymax])

    # 格式化X轴（百万为单位）
    def millions(x, pos):
        return '%1.1fM' % (x * 1e-6)

    formatter = FuncFormatter(millions)
    plt.gca().xaxis.set_major_formatter(formatter)

    # 设置图表样式
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('训练步骤', fontsize=18)
    plt.ylabel('平均回合奖励', fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()

    # 保存图表为PDF和PNG（PDF可能需要额外的软件查看）
    plt.savefig(os.path.join(plot_dir, 'ma2c_train.pdf'))
    plt.savefig(os.path.join(plot_dir, 'ma2c_train.png'))
    print(f"图表已保存至: {os.path.join(plot_dir, 'ma2c_train.pdf')}")
    print(f"图表已保存至: {os.path.join(plot_dir, 'ma2c_train.png')}")
    plt.close()


# Function for evaluation curve calculations - time-related operations
episode_sec = 3600


def fixed_agg(xs, window, agg):
    """
    Aggregate time-series data with fixed window size

    Args:
        xs: Values to aggregate
        window: Window size
        agg: Aggregation method ('sum', 'mean', or 'median')

    Returns:
        Aggregated values
    """
    xs = np.reshape(xs, (-1, window))
    if agg == 'sum':
        return np.sum(xs, axis=1)
    elif agg == 'mean':
        return np.mean(xs, axis=1)
    elif agg == 'median':
        return np.median(xs, axis=1)


def varied_agg(xs, ts, window, agg):
    """
    Aggregate time-series data with varied time intervals

    Args:
        xs: Values to aggregate
        ts: Timestamps
        window: Time window size
        agg: Aggregation method ('sum', 'mean', or 'median')

    Returns:
        Aggregated values
    """
    t_bin = window
    x_bins = []
    cur_x = []
    xs = list(xs) + [0]
    ts = list(ts) + [episode_sec + 1]
    i = 0
    while i < len(xs):
        x = xs[i]
        t = ts[i]
        if t <= t_bin:
            cur_x.append(x)
            i += 1
        else:
            if not len(cur_x):
                x_bins.append(0)
            else:
                if agg == 'sum':
                    x_stat = np.sum(np.array(cur_x))
                elif agg == 'mean':
                    x_stat = np.mean(np.array(cur_x))
                elif agg == 'median':
                    x_stat = np.median(np.array(cur_x))
                x_bins.append(x_stat)
            t_bin += window
            cur_x = []
    return np.array(x_bins)


def plot_series(df, name, tab, label, color, window=None, agg='sum', reward=False):
    """
    Plot a time series from evaluation data

    Args:
        df: DataFrame with data
        name: Column name to plot
        tab: Table type ('trip' or other)
        label: Series label
        color: Line color
        window: Window size for aggregation
        agg: Aggregation method
        reward: Whether plotting reward (affects formatting)

    Returns:
        Tuple of min and max y-values
    """
    episodes = list(df.episode.unique())
    num_episode = len(episodes)
    num_time = episode_sec
    print(label, name)
    # always use avg over episodes
    if tab != 'trip':
        res = df.loc[df.episode == episodes[0], name].values
        for episode in episodes[1:]:
            res += df.loc[df.episode == episode, name].values
        res = res / num_episode
        print('mean: %.2f' % np.mean(res))
        print('std: %.2f' % np.std(res))
        print('min: %.2f' % np.min(res))
        print('max: %.2f' % np.max(res))
    else:
        res = []
        for episode in episodes:
            res += list(df.loc[df.episode == episode, name].values)

        print('mean: %d' % np.mean(res))
        print('max: %d' % np.max(res))

    if reward:
        num_time = 720
    if window and (agg != 'mv'):
        num_time = num_time // window
    x = np.zeros((num_episode, num_time))
    for i, episode in enumerate(episodes):
        t_col = 'arrival_sec' if tab == 'trip' else 'time_sec'
        cur_df = df[df.episode == episode].sort_values(t_col)
        if window and (agg == 'mv'):
            cur_x = cur_df[name].rolling(window, min_periods=1).mean().values
        else:
            cur_x = cur_df[name].values
        if window and (agg != 'mv'):
            if tab == 'trip':
                cur_x = varied_agg(cur_x, df[df.episode == episode].arrival_sec.values, window, agg)
            else:
                cur_x = fixed_agg(cur_x, window, agg)
        x[i] = cur_x

    if num_episode > 1:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
    else:
        x_mean = x[0]
        x_std = np.zeros(num_time)

    if (not window) or (agg == 'mv'):
        t = np.arange(1, episode_sec + 1)
        if reward:
            t = np.arange(5, episode_sec + 1, 5)
    else:
        t = np.arange(window, episode_sec + 1, window)

    plt.plot(t, x_mean, color=color, linewidth=3, label=label)
    if num_episode > 1:
        x_lo = x_mean - x_std
        if not reward:
            x_lo = np.maximum(x_lo, 0)
        x_hi = x_mean + x_std
        plt.fill_between(t, x_lo, x_hi, facecolor=color, edgecolor='none', alpha=0.1)
        return np.nanmin(x_mean - 0.5 * x_std), np.nanmax(x_mean + 0.5 * x_std)
    else:
        return np.nanmin(x_mean), np.nanmax(x_mean)


def plot_combined_series(dfs, agent_names, col_name, tab_name, agent_labels, y_label, fig_name,
                         window=None, agg='sum', reward=False):
    """
    Plot combined time series for multiple agents

    Args:
        dfs: Dictionary of DataFrames
        agent_names: List of agent names
        col_name: Column name to plot
        tab_name: Table type
        agent_labels: List of agent labels
        y_label: Y-axis label
        fig_name: Output filename
        window: Window size for aggregation
        agg: Aggregation method
        reward: Whether plotting reward
    """
    plt.figure(figsize=(9, 6))
    ymin = np.inf
    ymax = -np.inf

    for i, aname in enumerate(agent_names):
        df = dfs[aname][tab_name]
        y0, y1 = plot_series(df, col_name, tab_name, agent_labels[i], COLORS[aname], window=window, agg=agg,
                             reward=reward)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    plt.xlim([0, episode_sec])
    if (col_name == 'average_speed') and ('global' in agent_names):
        plt.ylim([0, 6])
    elif (col_name == 'wait_sec') and ('global' not in agent_names):
        plt.ylim([0, 3500])
    else:
        plt.ylim([ymin, ymax])

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Simulation time (sec)', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.savefig(plot_dir + ('/%s.pdf' % fig_name))
    plt.close()


def sum_reward(x):
    """
    Sum reward from comma-separated string

    Args:
        x: Reward string

    Returns:
        Sum of rewards
    """
    x = [float(i) for i in x.split(',')]
    return np.sum(x)


def plot_eval_curve(scenario='large_grid', date='dec16'):
    """
    Plot evaluation curves for different algorithms

    Args:
        scenario: Environment name
        date: Date directory prefix
    """
    cur_dir = base_dir + ('/eval_%s/%s/eva_data' % (date, scenario))

    # Select algorithms to compare
    # names = ['ma2c', 'ia2c', 'iqll', 'greedy']
    # labels = ['MA2C', 'IA2C', 'IQL-LR', 'Greedy']
    # names = ['iqld', 'greedy']
    # labels = ['IQL-DNN','Greedy']
    names = ['ia2c', 'greedy']
    labels = ['IA2C', 'Greedy']

    dfs = {}
    for file in os.listdir(cur_dir):
        if not file.endswith('.csv'):
            continue
        if not file.startswith(scenario):
            continue
        name = file.split('_')[2]
        measure = file.split('_')[3].split('.')[0]
        if name in names:
            df = pd.read_csv(cur_dir + '/' + file)
            if name not in dfs:
                dfs[name] = {}
            dfs[name][measure] = df

    # plot avg queue
    plot_combined_series(dfs, names, 'avg_queue', 'traffic', labels,
                         'Average queue length (veh)', scenario + '_queue', window=60, agg='mv')
    # plot avg speed
    plot_combined_series(dfs, names, 'avg_speed_mps', 'traffic', labels,
                         'Average car speed (m/s)', scenario + '_speed', window=60, agg='mv')
    # plot avg waiting time
    plot_combined_series(dfs, names, 'avg_wait_sec', 'traffic', labels,
                         'Average intersection delay (s/veh)', scenario + '_wait', window=60, agg='mv')
    # plot trip completion
    plot_combined_series(dfs, names, 'number_arrived_car', 'traffic', labels,
                         'Trip completion rate (veh/5min)', scenario + '_tripcomp', window=300, agg='sum')
    # plot trip waiting time
    plot_combined_series(dfs, names, 'wait_sec', 'trip', labels,
                         'Avg trip delay (s)', scenario + '_tripwait', window=60, agg='mean')
    # plot rewards
    plot_combined_series(dfs, names, 'reward', 'control', labels,
                         'Step reward', scenario + '_reward', reward=True, window=6, agg='mv')


def get_mfd_points(df, dt_sec=300):
    """
    Calculate macroscopic fundamental diagram points

    Args:
        df: DataFrame with traffic data
        dt_sec: Time interval

    Returns:
        Tuple of outputs and accumulations
    """
    outputs = []
    accs = []
    ts = np.arange(0, 3601, dt_sec)
    for episode in df.episode.unique():
        cur_df = df[df.episode == episode]
        for i in range(len(ts) - 1):
            cur_df1 = cur_df[(cur_df.time_sec >= ts[i]) & (cur_df.time_sec < ts[i + 1])]
            outputs.append(np.sum(cur_df1.number_arrived_car.values) * 60 / dt_sec)
            accs.append(np.mean(cur_df1.number_total_car.values))
    return np.array(outputs), np.array(accs)


def plot_mfd_curve(scenario='real_net', date='oct07'):
    """
    Plot macroscopic fundamental diagram

    Args:
        scenario: Environment name
        date: Date directory prefix
    """
    cur_dir = base_dir + ('/eval_%s/%s/eva_data' % (date, scenario))
    names = ['ma2c', 'ia2c', 'greedy']
    labels = ['MA2C', 'IA2C', 'Greedy']
    dfs = {}

    for file in os.listdir(cur_dir):
        if not file.endswith('traffic.csv'):
            continue
        if not file.startswith(scenario):
            continue
        name = file.split('_')[2]
        if name not in names:
            continue
        df = pd.read_csv(cur_dir + '/' + file)
        outputs, accs = get_mfd_points(df)
        dfs[name] = (accs, outputs)

    plt.figure(figsize=(9, 6))
    styles = 'o^s'
    for i, name in enumerate(names):
        plt.scatter(dfs[name][0], dfs[name][1], s=80, marker=styles[i], c=COLORS[name],
                    edgecolors='none', label=labels[i], alpha=0.75)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Accumulation (veh)', fontsize=18)
    plt.ylabel('Output flow (veh/min)', fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.tight_layout()
    plt.savefig(plot_dir + ('/real_net_mfd.pdf'))
    plt.close()


def plot_flow_profile():
    """
    Plot traffic flow profiles
    """
    peak_flow1 = 1100
    peak_flow2 = 925
    colors = 'brgm'
    ratios1 = np.array([0.4, 0.7, 0.9, 1.0, 0.75, 0.5, 0.25])  # start from 0
    ratios2 = np.array([0.3, 0.8, 0.9, 1.0, 0.8, 0.6, 0.2])  # start from 15min
    flows1 = peak_flow1 * 0.6 * ratios1
    flows2 = peak_flow1 * ratios1
    flows3 = peak_flow2 * 0.6 * ratios2
    flows4 = peak_flow2 * ratios2
    flows = [list(flows1) + [0] * 6, list(flows2) + [0] * 6,
             [0] * 3 + list(flows3) + [0] * 3, [0] * 3 + list(flows4) + [0] * 3]
    t = np.arange(0, 3601, 300)
    t1 = t[:8]
    t2 = t[3:12]
    ts = [t1, t1, t2, t2]

    plt.figure(figsize=(9, 6))
    labels = ['f1', 'F1', 'f2', 'F2']
    for i in range(4):
        if i % 2 == 0:
            plt.step(t, flows[i], where='post', color=colors[i], linestyle=':', linewidth=3, label=labels[i])
        else:
            plt.step(t, flows[i], where='post', color=colors[i], linewidth=6, label=labels[i], alpha=0.5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Simulation time (sec)', fontsize=18)
    plt.ylabel('Flow rate (veh/hr)', fontsize=18)
    plt.legend(loc='best', fontsize=18)
    plt.xlim([0, 3600])
    plt.tight_layout()
    plt.savefig(plot_dir + ('/large_grid_flow.pdf'))
    plt.close()


# 主程序执行
if __name__ == "__main__":
    # 只运行MA2C训练曲线绘制
    plot_train_curve()
    print("程序执行完毕!")