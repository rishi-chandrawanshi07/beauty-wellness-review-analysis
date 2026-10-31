"""
Week 2 - Initial data profiling for the Sephora Products and Skincare Reviews dataset.

Usage (from the project root):
    python week2_data_profiling.py --products data/raw/product_info.csv \
                                   --reviews  data/raw/reviews_0-250.csv

Outputs:
    - a text profile printed to the console and saved to outputs/profile.txt
    - charts saved in outputs/figures/

Column names follow the Kaggle files; the script checks that each column exists
before using it, so it will not crash if a column name differs.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def has(df, *cols):
    return all(c in df.columns for c in cols)


def profile(df, name, log):
    """Print shape, dtypes, missing values, duplicates and summary statistics."""
    log(f"\n===== {name} =====")
    log(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    log(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    log(f"Fully duplicated rows: {df.duplicated().sum():,}")
    miss = df.isna().sum()
    miss = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)})
    log("\nMissing values (top 10 columns):")
    log(miss.sort_values("pct", ascending=False).head(10).to_string())
    log("\nData types:")
    log(df.dtypes.astype(str).value_counts().to_string())
    num = df.select_dtypes("number")
    if not num.empty:
        log("\nDescriptive statistics (numeric columns):")
        log(num.describe().T.round(2).to_string())


def save(fig_dir, fname):
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, fname), dpi=150)
    plt.close()


def product_charts(p, fig_dir, log):
    if has(p, "rating"):
        plt.figure(figsize=(7, 4))
        sns.histplot(p["rating"].dropna(), bins=20)
        plt.title("Distribution of average product ratings (Sephora catalogue)")
        plt.xlabel("Average rating (1-5)")
        plt.ylabel("Number of products")
        save(fig_dir, "fig1_product_rating_hist.png")
    if has(p, "price_usd"):
        q99 = p["price_usd"].quantile(0.99)
        plt.figure(figsize=(7, 4))
        sns.histplot(p.loc[p["price_usd"] <= q99, "price_usd"], bins=40)
        plt.title("Distribution of product prices (up to the 99th percentile)")
        plt.xlabel("Price (USD)")
        plt.ylabel("Number of products")
        save(fig_dir, "fig2_price_hist.png")
        log(f"\nPrice: median={p['price_usd'].median():.2f}, "
            f"mean={p['price_usd'].mean():.2f}, 99th pct={q99:.2f}")
    if has(p, "primary_category"):
        top = p["primary_category"].value_counts().head(10)
        log("\nProducts per primary category:\n" + top.to_string())
        plt.figure(figsize=(7, 4))
        sns.barplot(x=top.values, y=top.index)
        plt.title("Number of products by primary category")
        plt.xlabel("Number of products")
        plt.ylabel("Primary category")
        save(fig_dir, "fig3_category_counts.png")
    if has(p, "price_usd", "rating"):
        r, pval = (np.nan, np.nan)
        try:
            from scipy.stats import spearmanr
            sub = p[["price_usd", "rating"]].dropna()
            r, pval = spearmanr(sub["price_usd"], sub["rating"])
        except ImportError:
            pass
        log(f"\nSpearman correlation price vs rating: rho={r:.3f}, p={pval:.4g}")


def review_charts(r, fig_dir, log):
    if has(r, "rating"):
        counts = r["rating"].value_counts().sort_index()
        log("\nReview rating distribution:\n" + (counts / counts.sum() * 100).round(2).to_string())
        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("Number of reviews by star rating")
        plt.xlabel("Star rating")
        plt.ylabel("Number of reviews")
        save(fig_dir, "fig4_review_rating_counts.png")
    if has(r, "is_recommended"):
        share = r["is_recommended"].value_counts(normalize=True, dropna=False) * 100
        log("\nShare of is_recommended (%):\n" + share.round(2).to_string())
    if has(r, "rating", "is_recommended"):
        ct = pd.crosstab(r["rating"], r["is_recommended"], normalize="index") * 100
        log("\n% recommended within each star rating:\n" + ct.round(1).to_string())
    if has(r, "review_text"):
        words = r["review_text"].dropna().astype(str).str.split().str.len()
        log(f"\nReview length (words): median={words.median():.0f}, "
            f"mean={words.mean():.1f}, max={words.max()}")
        plt.figure(figsize=(7, 4))
        sns.histplot(words[words <= words.quantile(0.99)], bins=40)
        plt.title("Distribution of review length (words, up to 99th percentile)")
        plt.xlabel("Words per review")
        plt.ylabel("Number of reviews")
        save(fig_dir, "fig5_review_length.png")
    if has(r, "skin_type", "rating"):
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=r, x="skin_type", y="rating")
        plt.title("Review ratings by reviewer skin type")
        plt.xlabel("Skin type")
        plt.ylabel("Star rating")
        save(fig_dir, "fig6_rating_by_skin_type.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(str(msg))

    products = pd.read_csv(a.products)
    reviews = pd.read_csv(a.reviews, low_memory=False)

    profile(products, "PRODUCTS", log)
    profile(reviews, "REVIEWS", log)
    product_charts(products, fig_dir, log)
    review_charts(reviews, fig_dir, log)

    if has(products, "product_id") and has(reviews, "product_id"):
        matched = reviews["product_id"].isin(products["product_id"]).mean() * 100
        log(f"\nReviews whose product_id exists in the product table: {matched:.2f}%")

    with open(os.path.join(a.out, "profile.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved profile to {a.out}/profile.txt and charts to {fig_dir}/")


if __name__ == "__main__":
    main()
