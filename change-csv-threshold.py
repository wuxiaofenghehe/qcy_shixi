import pandas as pd

# 读取 CSV
df = pd.read_csv(r"E:\qcy\csv-document\threshold_csv_test.csv")

# 只修改 gdf 和 gzf 的 confidence
df.loc[df["type"].isin(["gdf", "gzf"]), "confidence"] = 0.8

# 保存为新文件（避免覆盖原文件）
df.to_csv(r"E:\qcy\csv-document\threshold_csv_test_0.8.csv", index=False)

print("修改完成!")
