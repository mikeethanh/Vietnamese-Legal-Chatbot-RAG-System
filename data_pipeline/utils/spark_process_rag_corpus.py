#!/usr/bin/env python3
"""
utils/spark_process_rag_corpus.py

Usage:
spark-submit \
  --master local[*] \
  --driver-memory 12g \
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle \
  utils/spark_process_rag_corpus.py \
    --raw-prefix s3a://legal-datalake/raw/rag_corpus \
    --out-prefix s3a://legal-datalake/process/rag_corpus \
    --coalesce
"""

import argparse
import sys
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, trim, md5, concat

# -------------------
# Spark init
# -------------------
def build_spark():
    builder = (
        SparkSession.builder
        .appName("ProcessLegalCorpusJSONL")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Nếu AWS credentials chưa set trong env ~/.aws/credentials, uncomment dòng dưới
        # .config("spark.hadoop.fs.s3a.access.key", "YOUR_AWS_ACCESS_KEY")
        # .config("spark.hadoop.fs.s3a.secret.key", "YOUR_AWS_SECRET_KEY")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
    )
    return builder.getOrCreate()

# -------------------
# Read CSV / JSON
# -------------------
def read_csv_text(spark, path, text_col):
    try:
        df = spark.read.option("header", "true").csv(path)
        if text_col not in df.columns:
            print(f"⚠️ CSV {path} missing column {text_col}")
            return None
        return df.select(col(text_col).cast("string").alias("text")).withColumn("origin", lit(path.split("/")[-1]))
    except Exception as e:
        print(f"⚠️ Failed to read CSV {path}: {e}")
        return None

def safe_read_json(spark, path, multiline=True):
    try:
        df = spark.read.option("multiline", "true").json(path)
        return df
    except Exception as e:
        print(f"⚠️ Failed to read JSON {path}: {e}")
        return None

def extract_from_articles(df, src_prefix):
    try:
        exploded = df.select(explode(col("articles")).alias("article"))
        out = exploded.select(col("article.text").cast("string").alias("text"))
        out = out.withColumn("origin", lit(src_prefix))
        return out
    except Exception as e:
        print(f"⚠️ extract_from_articles failed: {e}")
        return None

def extract_text_generic(df, src_prefix):
    if df is None:
        return None
    cols = df.columns
    if "text" in cols:
        return df.select(col("text").cast("string").alias("text")).withColumn("origin", lit(src_prefix))
    if "content" in cols:
        return df.select(col("content").cast("string").alias("text")).withColumn("origin", lit(src_prefix))
    if "articles" in cols:
        return extract_from_articles(df, src_prefix)
    print(f"⚠️ No 'text'/'content'/'articles' in {src_prefix}")
    return None

def build_id_expr():
    return md5(concat(lit("src_"), col("origin"), lit("_"), col("text")))

# -------------------
# Main processing
# -------------------
def main(args):
    spark = build_spark()
    
    raw_base = args.raw_prefix.rstrip("/")
    out_base = args.out_prefix.rstrip("/")

    print(f"📂 Processing legal corpus from: {raw_base}")
    print(f"📁 Output will be written to: {out_base}")
    print(f"🏗️  Spark App: {spark.sparkContext.appName}")

    sources = []
    total_records = 0

    # List of CSV files (tên phải khớp với S3)
    files_to_process = [
        ("corpus.csv", "text"),
        ("data (1).csv", "full_text"),
        ("updated_legal_corpus.csv", "content")
    ]

    # List of JSON files
    json_files = ["legal_corpus.json", "zalo_corpus.json", "vbpl_crawl.json"]

    # Process CSV
    for filename, text_col in files_to_process:
        file_path = f"{raw_base}/{filename}"
        print(f"\n📄 Processing {file_path} ...")
        df = read_csv_text(spark, file_path, text_col)
        if df is not None:
            count = df.count()
            sources.append(df)
            total_records += count
            print(f"✅ {filename} - {count:,} records")
        else:
            print(f"⚠️  Skipped {filename}")

    # Process JSON
    for json_file in json_files:
        file_path = f"{raw_base}/{json_file}"
        print(f"\n📄 Processing {file_path} ...")
        raw_df = safe_read_json(spark, file_path, multiline=True)
        text_df = extract_text_generic(raw_df, json_file)
        if text_df is not None:
            count = text_df.count()
            sources.append(text_df)
            total_records += count
            print(f"✅ {json_file} - {count:,} records")
        else:
            print(f"⚠️  Skipped {json_file}")

    print(f"\n📊 Summary: Found {len(sources)} sources with {total_records:,} total records")

    if not sources:
        print("❌ No valid sources. Exiting.")
        spark.stop()
        sys.exit(1)

    # Union all
    df_union = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), sources)

    # Clean text
    df_clean = df_union.withColumn("text", trim(col("text"))).filter((col("text").isNotNull()) & (col("text") != ""))
    clean_count = df_clean.count()
    print(f"📈 After cleaning: {clean_count:,} valid records")

    # Add deterministic ID
    df_final = df_clean.withColumn("id", build_id_expr()).select("id", "text", "origin")
    df_final = df_final.dropDuplicates(["id"])
    final_count = df_final.count()
    print(f"📊 Final dataset: {final_count:,} unique records")

    # Coalesce if needed
    if args.coalesce:
        df_final = df_final.coalesce(1)

    # Save JSONL to S3
    final_object_path = f"{out_base}/combined.jsonl"
    df_final.select("id", "text").write.mode("overwrite").json(final_object_path)
    print(f"✅ Combined JSONL written to {final_object_path}")

    spark.stop()
    print("All done.")

# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Process Vietnamese legal corpus using Spark + S3")
    p.add_argument("--raw-prefix", default="s3a://legal-datalake/raw/rag_corpus")
    p.add_argument("--out-prefix", default="s3a://legal-datalake/process/rag_corpus")
    p.add_argument("--coalesce", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print("🏛️  VIETNAMESE LEGAL CORPUS PROCESSING")
    print("=" * 60)
    print(f"📂 Raw data path: {args.raw_prefix}")
    print(f"📁 Output path: {args.out_prefix}")
    print(f"📦 Coalesce output: {args.coalesce}")
    print("=" * 60)

    try:
        main(args)
        print("\n🎉 Legal corpus processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)
