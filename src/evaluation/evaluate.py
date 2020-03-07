import os

from src.evaluation.metrics_from_log import TrainingLogEvaluator
from src.evaluation.metrics_from_log import LogStatisticsCalculator


def plot_analysis(dir_, show=False):
    for item in os.scandir(dir_):
        if item.name == "logs.txt":
            tle = TrainingLogEvaluator(item.path)
            item_dir = os.path.dirname(item.path)
            tle.plot_classwise_prediction_bars(
                save_name=os.path.join(item_dir, "correct_counts.png"),
                ylim=[0, 1000],
                show=show
            )
            tle.plot_classwise_prediction_bars(
                mask=[0]+[1]*31,
                save_name=os.path.join(item_dir, "correct_counts_wo_0.png"),
                ylim=[0, 150],
                show=show
            )
            tle.plot_classwise_correct_and_incorrect_bars(
                save_name=os.path.join(
                    item_dir, "correct_incorrect_counts.png"),
                ylim=[0, 1000],
                show=show
            )
            tle.plot_classwise_correct_and_incorrect_bars(
                mask=[0]+[1]*31,
                save_name=os.path.join(
                    item_dir, "correct_incorrect_counts_wo_0.png"),
                ylim=[0, 150],
                show=show
            )
        elif item.is_dir():
            plot_analysis(item.path)
        else:
            continue


def statistics_analysis(dir_):
    tles = list()
    for item in os.scandir(dir_):
        try:
            tle = TrainingLogEvaluator(os.path.join(item.path, "logs.txt"))
            tles.append(tle)
        except FileNotFoundError:
            continue
    lsc = LogStatisticsCalculator(
        tles,
        output_path=os.path.join(dir_, "statistics.log"))
    lsc.statistics_str()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the log folder.")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show the plottings.")
    args = parser.parse_args()
    plot_analysis(args.path, args.show)
    statistics_analysis(args.path)
