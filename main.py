from BigBrother import *


mat = Material(E=2.59e9, nu=0.35)
files = select_files()

with plt.style.context("ggplot"):
    for file in files:
        # df = read_with_polars(
        #     file,
        #     separator=";",
        #     skip_rows=7,
        #     skip_rows_after_header=1
        # )
        df = read_with_pandas(
            file,
            sep=";",
            skiprows=list(range(7))+[8],
        )
        print(df)
        BigBrother(df, mat, -5000e-6)
        # MeanBrother(df, mat, -5000e-6)
        plt.show()