import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


def main(split_number):
    """Main function to run regression with TabPFN and conformal prediction."""
    train_data = np.load(f"real_world_data/bio/data/bio_train_split_{split_number}.npz")
    X_train, y_train = train_data["X"], train_data["y"]

    test_data = np.load(f"real_world_data/bio/data/bio_test_split_{split_number}.npz")
    X_test, y_test = test_data["X"], test_data["y"]

    cal_data = np.load(f"real_world_data/bio/data/bio_cal_split_{split_number}.npz")
    X_cal, y_cal = cal_data["X"], cal_data["y"]

    # Initialize the regressor
    regressor = TabPFNRegressor().create_default_for_version(ModelVersion.V2)
    regressor.fit(X_train, y_train)

    # Predict on the test set
    predictions = regressor.predict(X_test, output_type="main", quantiles=[0.05, 0.95])

    q1_test = predictions["quantiles"][0]
    q2_test = predictions["quantiles"][1]
    preds = predictions["mean"]

    # Conformalization
    cal_pred = regressor.predict(X_cal, output_type="main", quantiles=[0.05, 0.95])
    q1_cal = cal_pred["quantiles"][0]
    q2_cal = cal_pred["quantiles"][1]

    s1 = q1_cal - y_cal.squeeze()
    s2 = y_cal.squeeze() - q2_cal
    scores = np.maximum(s1, s2)

    alpha = 0.1
    n = len(X_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    q_alpha_hat = np.sort(scores)[k - 1]

    # Final band
    print(f"q_alpha_hat: {q_alpha_hat}")
    q1 = q1_test - q_alpha_hat
    q2 = q2_test + q_alpha_hat

    # Compute metrics
    coverage = (q1 <= y_test.squeeze()) & (y_test.squeeze() <= q2)
    mean_width = np.mean(q2 - q1)
    print(f"Mean width: {mean_width}")
    print(f"Maringal coverage: {np.mean(coverage)}")

    rc_nclust = 10
    rc_min = 100

    wsc = []
    for i in range(10):
        rng = np.random.default_rng(seed=i)
        centers = rng.choice(X_test, rc_nclust, replace=False)

        regional_coverages = []
        for clust, point in enumerate(centers):
            norms_ranks = np.linalg.norm(point - X_test, axis=1)
            order = np.argsort(norms_ranks)
            regional_coverage = np.mean(coverage[order[:rc_min]])
            regional_coverages.append(regional_coverage)

        worst_set_coverage = np.array(regional_coverages).min()
        wsc.append(worst_set_coverage)

        # Build a tidy DataFrame
        df = pd.DataFrame(
            {
                "wsc": wsc,
                "method": "TabPFN_v2",  # same label for all rows
            }
        )

        sns.set_theme(style="ticks")

        f, ax = plt.subplots(figsize=(7, 3))

        sns.boxplot(
            data=df,
            x="wsc",
            y="method",
            whis=(0, 100),
            width=0.6,
            showcaps=False,
            showfliers=False,
            boxprops={"facecolor": "none"},  # hollow box, just outline
            medianprops={"color": "black"},
            ax=ax,
        )

        sns.stripplot(
            data=df,
            x="wsc",
            y="method",
            size=5,
            color=".3",
            alpha=0.8,
            ax=ax,
        )

        mean_wsc = df["wsc"].mean()
        ax.scatter(mean_wsc, 0, marker="s", s=80, color="black", zorder=5)
        ax.set_title(
            f"Mean width: {mean_width:.2f}, Marginal coverage: {np.mean(coverage):.3f}, Worst-set coverage: {mean_wsc:.3f}"
        )
        # Cosmetics
        ax.xaxis.grid(True)
        ax.set(xlabel="Worst-set coverage", ylabel="")
        sns.despine(trim=True, left=True)

        plt.tight_layout()
        plt.savefig(f"worse-set-coverage_split_{split_number}.png")


if __name__ == "__main__":
    """Run main for different splits."""
    for split_number in range(10):
        main(split_number)
