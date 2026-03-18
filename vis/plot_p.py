from __future__ import annotations
import argparse
import os
import sys
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 17,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
    "figure.titlesize": 17,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot longitudinal RapCluster summary trends from 2000 to 2025."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="rapcluster_longitudinal_summary.png",
        help="Output image path (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: %(default)s)",
    )
    return parser.parse_args()


def validate_lengths(series: dict[str, list[float]], years: list[int]) -> None:
    n_years = len(years)
    for name, values in series.items():
        if len(values) != n_years:
            raise ValueError(
                f"Series '{name}' has length {len(values)}, expected {n_years}"
            )


def main() -> int:
    args = parse_args()

    years = list(range(2000, 2026))

    total_articles = [
        300, 369, 435, 568, 882, 1310, 1972, 2908, 4132, 5308, 7441, 9616, 12740,
        16321, 19537, 22918, 26414, 30401, 35135, 42034, 56349, 73084, 86707,
        81279, 87698, 110541,
    ]

    any_algorithm_n = [
        272, 338, 415, 535, 831, 1197, 1792, 2619, 3746, 4776, 6556, 8509, 11246,
        14523, 17484, 20395, 23325, 27033, 31249, 37411, 50230, 65284, 77164,
        72595, 79463, 102928,
    ]

    any_algorithm_pct = [
        90.66666666666666, 91.59891598915989, 95.40229885057471, 94.19014084507043,
        94.21768707482994, 91.37404580152672, 90.87221095334685, 90.06189821182944,
        90.65827686350435, 89.97739261492087, 88.1064373068136, 88.4879367720466,
        88.27315541601256, 88.98351816677899, 89.49173363361827, 88.9911859673619,
        88.30544408268342, 88.92141705864938, 88.93980361462928, 89.00176047961175,
        89.14088981170917, 89.32734935143122, 88.99396819172615, 89.31581343274401,
        90.60982006431162, 93.11296261115784,
    ]

    missing_reporting_pct = [
        100.0, 99.1869918699187, 99.08045977011494, 99.29577464788733,
        99.65986394557824, 99.16030534351144, 98.93509127789046, 98.55570839064649,
        99.05614714424009, 98.9638281838734, 98.95175379653273, 99.07445923460898,
        99.11302982731554, 98.95839715703694, 98.86881302144648, 98.92660790644908,
        98.85288104792913, 98.63162395973816, 98.50860964849865, 98.49407622400913,
        98.34247280342153, 98.13091784795577, 98.15701154462731, 97.81862473701695,
        97.625943579101, 96.8247075745651,
    ]

    missing_parameters_pct = [
        98.66666666666667, 96.4769647696477, 90.57471264367815, 91.90140845070422,
        87.64172335600907, 87.25190839694657, 87.67748478701826, 85.07565337001375,
        85.26137463697968, 86.98417483044461, 86.52062894772208, 88.17387687187915,
        87.45682888540031, 87.26793701366362, 86.61309208169115, 86.43249847281525,
        86.30953206632839, 85.84684648531298, 84.98363455244158, 84.84084217538279,
        84.44160499831408, 83.17797328921898, 82.04158983415365, 81.65024287035409,
        81.21279846758283, 80.15578292036131,
    ]

    missing_justification_pct = [
        37.666666666666664, 38.21138211382114, 37.93103448275862, 42.07746478873239,
        38.095238095238095, 35.267175572519086, 34.17849898580122, 31.73933975240715,
        30.614714424007746, 30.78372268274303, 31.097970702862518, 30.949459234608985,
        31.059654631083204, 28.889161203357635, 29.52346829093515, 29.757396980539314,
        28.719618384947376, 28.07506332061445, 28.030453963284475, 27.00147499643146,
        28.174413920388117, 27.405454545454546, 26.991131857180277, 25.338522987733606,
        23.77249196093411, 22.482157393178643,
    ]

    missing_evaluation_pct = [
        86.0, 80.75880758807588, 84.13793103448276, 84.33098591549296,
        85.14739229024943, 84.42748091603053, 82.5050709939148, 80.05495185694635,
        79.93610842207164, 79.85983345893066, 80.64803117860503, 80.77267054804597,
        80.03139717425431, 79.22798848109798, 79.07457644571839, 78.71995811152806,
        78.66131672673582, 78.23755830400315, 77.69531805905792, 77.40686111243279,
        77.31636763740261, 76.45421651319003, 76.43258271984915, 75.79325594532277,
        74.45095669228499, 71.75392396811825,
    ]

    missing_tuning_pct = [
        93.33333333333333, 94.3089430894309, 92.87356321839081, 91.02112676056338,
        91.49659863945578, 88.85496183206106, 87.67748478701826, 87.3796423658872,
        87.39012584704743, 87.60361718161267, 88.60368230130359, 89.00062499999999,
        88.61852329042387, 87.89167354941487, 88.68198802272508, 88.11152805654944,
        87.6345877186333, 87.13529160159206, 86.38793226127807, 85.63543893038968,
        85.31473495447922, 84.53040528569386, 83.50254316503004, 82.45427429053187,
        81.30288033820612, 78.284894287468,
    ]

    percent_series = {
        "Any algorithm match": any_algorithm_pct,
        "Missing reporting signals": missing_reporting_pct,
        "Missing parameters": missing_parameters_pct,
        "Missing justification": missing_justification_pct,
        "Missing evaluation": missing_evaluation_pct,
        "Missing tuning": missing_tuning_pct,
    }

    validate_lengths(
        {
            "total_articles": total_articles,
            "any_algorithm_n": any_algorithm_n,
            **percent_series,
        },
        years,
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.7]},
        constrained_layout=True,
    )
    
    fig.set_constrained_layout_pads(hspace=0.08, h_pad=0.02, w_pad=0.02)

    ax1.plot(
        years,
        total_articles,
        linewidth=2.8,
        marker="o",
        markersize=4,
        label="Total articles",
    )
    ax1.set_ylabel("Articles per year", fontsize = 25, fontweight="bold", fontname="Arial")
    for label in ax1.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight("bold")
        label.set_fontname("Arial")
    #ax1.set_title("Longitudinal summary of clustering-method reporting (2000–2025)")
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(
        years,
        any_algorithm_pct,
        linewidth=3.0,
        label="Any algorithm match",
    )
    ax2.plot(
        years,
        missing_reporting_pct,
        linewidth=3.0,
        label="Missing reporting signals",
    )

    ax2.plot(
        years,
        missing_parameters_pct,
        linewidth=1.8,
        linestyle="--",
        label="Missing parameters",
    )
    ax2.plot(
        years,
        missing_justification_pct,
        linewidth=1.8,
        linestyle="--",
        label="Missing justification",
    )
    ax2.plot(
        years,
        missing_evaluation_pct,
        linewidth=1.8,
        linestyle="--",
        label="Missing evaluation",
    )
    ax2.plot(
        years,
        missing_tuning_pct,
        linewidth=1.8,
        linestyle="--",
        label="Missing tuning",
    )
    ax2.set_xlabel("Year", fontsize = 25, fontweight = "bold", fontname = "Arial")
    ax2.set_ylabel("Articles (%)", fontsize = 25, fontweight = "bold", fontname = "Arial")
    for label in ax2.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight("bold")
        label.set_fontname("Arial")
    for label in ax2.get_xticklabels():
        label.set_fontsize(20)
        label.set_fontweight("bold")
        label.set_fontname("Arial")
    ax2.set_xticks(list(range(2000, 2026, 5)))
    ax2.grid(True, alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        prop={'size': 20, 'weight': 'bold', 'family': 'Arial'}
    )

    ax1.text(
        0.01,
        1.03,
        "A",
        transform=ax1.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
    )

    ax2.text(
        0.01,
        1.03,
        "B",
        transform=ax2.transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    pdf_output = os.path.splitext(args.output)[0] + ".pdf"
    fig.savefig(pdf_output, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())