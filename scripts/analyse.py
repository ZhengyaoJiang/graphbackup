import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from pathlib import Path
from scipy.stats import wilcoxon

parser = argparse.ArgumentParser(description="Analyse the logs produced by torchbeast")

parser.add_argument("--dir", type=str, default="~/locallogs/ava", help="Directory for log files.")
parser.add_argument("--mode", type=str, default="table", choices=["table", "plot", "joint_plot",
                                                                  "group_plot", "integrate_table", "integrate_plot",
                                                                  "states_portion"])
parser.add_argument("--idx", "--index", nargs="+", required=True)
parser.add_argument("--repeats", default=3, type=int)
parser.add_argument("--steps", default=float('inf'), type=float)
parser.add_argument("--baseline", default=0.0, type=float)
parser.add_argument("--labels",nargs="+", help="labels for differnt tasks")
parser.add_argument("--tasks",nargs="+", help="task names")
parser.add_argument("--task_masks", nargs="*", help="masks 0 means the task is no needed to be summarised")
parser.add_argument("--human_scores", nargs="*", help="scores of human expert performance")
parser.add_argument("--random_scores", nargs="*", help="scores generated by random play")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--summary", type=str, default="median")

def integrate_table(tasks, indexes, labels, dir, steps, name, repeats, summary, human_scores, random_scores, format="text"):
    data = {label:[] for label in labels}
    print(tasks)
    ht_data = {label:[] for label in labels}
    for group_n, (index, label) in enumerate(zip(indexes, labels)):
        for task_n, task in enumerate(tasks):
            mean_l, std_l = [], []
            for round in range(repeats):
                task_id, code = index.split("-")
                try:
                    df = pd.read_csv(os.path.join(dir, f"{task_id}-{int(code)+task_n}-{round+1}", "logs.csv"))
                    nb_mean = (len(df["mean_episode_return"])+1) // 10
                    if human_scores:
                        last = df["mean_episode_return"].iloc[-nb_mean:]# / human_scores[task_n] * 100
                        random_score = float(random_scores[task_n])
                        last = (last-random_score) / (float(human_scores[task_n])-random_score) *100
                    else:
                        last = df["mean_episode_return"].iloc[-nb_mean:]# / human_scores[task_n] * 100
                    ht_data[label].append(last.mean())
                    mean_l.append(last.mean())
                    std_l.append(last.std())
                except FileNotFoundError:
                    print(f"skip {task_id}-{int(code)+task_n}-{round+1}")
            mean_return = np.mean(mean_l)
            mean_std = np.mean(std_l)
            data[label].append(mean_return)
    df = pd.DataFrame(data, index=tasks)
    mean, median = df.mean(), df.median()
    df.loc["mean"] = mean
    df.loc["median"] = median
    if format=="text":
        print(df)
    print(f"results for wilcoxon test")
    for (method1, data1) in ht_data.items():
        for (method2, data2) in ht_data.items():
            if method1 != method2:
                _, p_value = wilcoxon(data1, data2, alternative="greater")
                print(f"the p value for {method1} is better than {method2} is {p_value}")


def integrate_plot(tasks, indexes, labels, dir, steps, name, repeats, summary, human_scores,
                   format="text", base_scores=0, task_masks=None):
    fig = plt.figure(figsize=(3.1, 3.0))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.gcf().subplots_adjust(bottom=0.17, left=0.19)
    for group_n, (index, label) in enumerate(zip(indexes, labels)):
        data = []
        data_std = []
        for task_n, task in enumerate(tasks):
            if isinstance(task_masks, list):
                if task_masks[task_n] == "0":
                    continue
            curves = {}
            for round in range(repeats):
                task_id, code = index.split("-")
                try:
                    df = pd.read_csv(os.path.join(dir, f"{task_id}-{int(code)+task_n}-{round+1}", "logs.csv"), index_col="step")
                    if human_scores:
                        returns = df["mean_episode_return"].ewm(span=1).mean()
                        if isinstance(base_scores, list):
                            random_scores = np.array(base_scores, dtype=np.float32)[task_n]
                        else:
                            random_scores = 0
                        returns = (returns-random_scores) / (float(human_scores[task_n])-random_scores) *100
                    else:
                        returns = df["mean_episode_return"].ewm(span=1).mean()
                    curves[round]=returns
                except FileNotFoundError:
                    print(f"skip {task_id}-{int(code) + task_n}-{round + 1}")
            curvesdf = pd.concat(list(curves.values()), axis=1)
            curvesdf = curvesdf[curvesdf.index <= steps]
            summary_curve = curvesdf.mean(axis=1)
            data.append(summary_curve)
            data_std.append(curvesdf.std(axis=1))
        if summary == "median":
            tasks_df = pd.concat(data, axis=1)
            tasks_summary = tasks_df.median(axis=1).values.transpose()
            tasks_std = pd.concat(data_std, axis=1)
            tasks_std = tasks_std.median(axis=1).values.transpose()
            plt.plot(curvesdf.index, tasks_summary, linewidth=1.5, label=label)
            plt.fill_between(curvesdf.index, tasks_summary-tasks_std,
                             tasks_summary-tasks_std, alpha=0.3)
        elif summary == "mean":
            tasks_df = pd.concat(data, axis=1)
            tasks_summary = tasks_df.mean(axis=1).values.transpose()
            tasks_std = pd.concat(data_std, axis=1)
            tasks_std = tasks_std.mean(axis=1).values.transpose()
            plt.plot(curvesdf.index, tasks_summary, linewidth=1.5, label=label)
            plt.fill_between(curvesdf.index, tasks_summary-tasks_std,
                             tasks_summary-tasks_std, alpha=0.3)
        else:
            raise ValueError()
        plt.xlabel('step')
        if human_scores:
            plt.ylabel(f'{summary} normalised score (%)')
        else:
            plt.ylabel(f'{summary} score')
        plt.legend(labels=labels, prop={'size': 7})
        plt.title(name.split("/")[-1].replace("_", " ").replace("-", " "))
        if not name:
            name = os.path.expanduser(os.path.join(dir, "".join(indexes) + "return.png"))
        else:
            name = os.path.expanduser(os.path.join(dir, name))
        Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
        plt.savefig(name)


def group_plot(indexes, labels, dir, steps, name, repeats=3, summary="mean", baseline=0.0):
    fig = plt.figure(figsize=(3.0, 3.0))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.gcf().subplots_adjust(bottom=0.17, left=0.15)
    summary_lines = []
    for group_n,(index, label) in enumerate(zip(indexes, labels)):
        in_group = []
        for i in range(repeats):
            try:
                df = pd.read_csv(os.path.join(dir, f"{index}-{i+1}", "logs.csv"))
                df = df.reset_index(drop=True)
                if "frames" in df.columns:
                    df["step"] = df["frames"]
                df = df[df["step"]<steps]
                df["label"] = label
                df["id"] = f"{index}-{i+1}"
                df["size"] = 0.1
                df["smoothed return"] = df["mean_episode_return"].ewm(span=10).mean()
                step = df["step"]
                returns = df["smoothed return"]
                plt.plot(step, returns, f"C{group_n}", alpha=0.3, linewidth=0.3)
                in_group.append([step, returns, np.array(returns)[-1]])
            except:
                print(f"skip{i}")
        if summary == "median":
            final_returns = [in_group[i][-1] for i in range(repeats)]
            median = in_group[np.argsort(final_returns)[len(final_returns) // 2]]
            summary_lines.append(plt.plot(median[0], median[1], f"C{group_n}", linewidth=1.5)[0])
        elif summary == "mean":
            mean = [np.mean(pd.DataFrame([x[0] for x in in_group]), axis=0),
                    np.mean(pd.DataFrame([x[1] for x in in_group]), axis=0)]
            summary_lines.append(plt.plot(mean[0], mean[1], f"C{group_n}", linewidth=1.5)[0])
    if baseline != 0.0:
        plt.axhline(baseline, color="r", linestyle="dashed")
    plt.xlabel('step')
    plt.ylabel('smoothed return')
    plt.legend(summary_lines, labels, prop={'size': 7})
    plt.title(name.split("/")[-1].replace("_"," ").replace("-", " "))
    if not name:
        name = os.path.expanduser(os.path.join(dir, "".join(indexes)+"return.png"))
    else:
        name = os.path.expanduser(os.path.join(dir, name))
    Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
    plt.savefig(name)


def plot_mean_return(index, dir):
    plt.figure()
    df = pd.read_csv(os.path.join(dir, index, "logs.csv"), index_col="# _tick")
    df["smoothed return"] = df["mean_episode_return"].ewm(span=100).mean()
    sns.relplot(x="step", y="smoothed return", kind="line", data=df, aspect=2.0)
    plt.savefig(os.path.expanduser(os.path.join(dir, index, "return.png")))

def compute_return_stats(indexes, dir):
    result = {}
    for i in indexes:
        df = pd.read_csv(os.path.join(dir,i,"logs.csv"), index_col="# _tick")
        last = df["mean_episode_return"].iloc[-10:]
        result[i]=(last.mean(), last.std())
    return result

def print_stats_dict(stats):
    for i, stat in stats.items():
        print(f"{i}: {stat[0]:.2f} ± {stat[1]:.2f}")

def parse_state_portions(tasks, indexes,
                         labels, dir, steps, name, repeats, summary, human_scores, random_scores):
    data = {label:[] for label in labels}
    print(tasks)
    for group_n, (index, label) in enumerate(zip(indexes, labels)):
        data[label+"new_states_portion"] = []
        for task_n, task in enumerate(tasks):
            mean_l, std_l = [], []
            portion = []
            for round in range(repeats):
                task_id, code = index.split("-")
                try:
                    df = pd.read_csv(os.path.join(dir, f"{task_id}-{int(code)+task_n}-{round+1}", "logs.csv"))
                    nb_mean = (len(df["mean_episode_return"])+1) // 10
                    if human_scores:
                        last = df["mean_episode_return"].iloc[-nb_mean:]# / human_scores[task_n] * 100
                        random_score = float(random_scores[task_n])
                        last = (last-random_score) / (float(human_scores[task_n])-random_score) *100
                    else:
                        last = df["mean_episode_return"].iloc[-nb_mean:]# / human_scores[task_n] * 100
                    portion.append(df["number_of_states"]/(df["step"]+1))
                    mean_l.append(last.mean())
                    std_l.append(last.std())
                except FileNotFoundError:
                    print(f"skip {task_id}-{int(code)+task_n}-{round+1}")
            mean_return = np.mean(mean_l)
            mean_std = np.mean(std_l)
            mean_portion = np.mean(portion)
            data[label].append(mean_return)
            data[label+"new_states_portion"].append(mean_portion)

    df = pd.DataFrame(data, index=tasks)
    mean, median = df.mean(), df.median()
    df["relative performance"] = df[labels[1]] / df[labels[0]]
    df.loc["mean"] = mean
    df.loc["median"] = median
    print(df)

def main(flags):
    if flags.mode == "table":
        print_stats_dict(compute_return_stats(flags.idx, flags.dir))
    if flags.mode == "plot":
        for idx in flags.idx:
            plot_mean_return(idx, flags.dir)
    if flags.mode == "group_plot":
        group_plot(flags.idx, flags.labels, flags.dir, flags.steps, flags.name,
                   repeats=flags.repeats, baseline=flags.baseline)
    if flags.mode == "integrate_table":
        integrate_table(flags.tasks, flags.idx, flags.labels, flags.dir, flags.steps, flags.name,
                        repeats=flags.repeats, human_scores=flags.human_scores, random_scores=flags.random_scores, summary=flags.summary)
    if flags.mode == "integrate_plot":
        integrate_plot(flags.tasks, flags.idx, flags.labels, flags.dir, flags.steps, flags.name,
                        repeats=flags.repeats, human_scores=flags.human_scores,
                        summary=flags.summary, task_masks=flags.task_masks, base_scores=flags.random_scores)
    if flags.mode == "states_portion":
        parse_state_portions(flags.tasks, flags.idx, flags.labels, flags.dir, flags.steps, flags.name,
                             repeats=flags.repeats, human_scores=flags.human_scores, random_scores=flags.random_scores,
                             summary=flags.summary)





if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
