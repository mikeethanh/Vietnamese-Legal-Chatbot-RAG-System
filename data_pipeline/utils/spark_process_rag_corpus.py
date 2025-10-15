#!/usr/bin/env python3
"""
utils/spark_process_rag_corpus.py

Usage (example):
spark-submit \
  --master local[*] \
  --driver-memory 12g \
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.11.901 \
  utils/spark_process_rag_corpus.py \
    --minio-endpoint http://localhost:9000 \
    --access-key minio_access_key \
    --secret-key minio_secret_key \
    --bucket datalake \
    --raw-prefix raw/rag_corpus \
    --out-prefix processed/rag_corpus \
    --coalesce
"""
import argparse
import sys
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, lit, trim, md5, concat, input_file_name
)

# -------------------
# Helper / Spark init
# -------------------
def build_spark(minio_endpoint, access_key, secret_key):
    return SparkSession.builder \
        .appName("ProcessRagCorpusJSONL") \
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint) \
        .config("spark.hadoop.fs.s3a.access.key", access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", secret_key) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

def safe_read_json(spark, path, multiline=True):
    """Try to read json with multiline first, else fallback to normal json (useful if jsonl)"""
    try:
        if multiline:
            df = spark.read.option("multiline", "true").json(path)
        else:
            df = spark.read.json(path)
        print(f"Read JSON (multiline={multiline}) path={path}, columns={df.columns}")
        return df
    except Exception as e:
        print(f"⚠️ multiline read failed for {path}: {e} -- trying non-multiline")
        try:
            df = spark.read.json(path)
            print(f"Read JSON non-multiline path={path}, columns={df.columns}")
            return df
        except Exception as e2:
            print(f"❌ Failed to read JSON at {path}: {e2}")
            return None

def read_csv_text(spark, path, text_col):
    try:
        df = spark.read.option("header", "true").csv(path)
        if text_col not in df.columns:
            print(f"⚠️ CSV at {path} does not contain column '{text_col}', available cols: {df.columns}")
            return None
        return df.select(col(text_col).cast("string").alias("text")).withColumn("origin", lit(path))
    except Exception as e:
        print(f"❌ Failed to read CSV {path}: {e}")
        return None

def extract_from_articles(df, src_prefix):
    # df expected to have column 'articles' which is an array of structs with 'text'
    try:
        exploded = df.select(explode(col("articles")).alias("article"))
        out = exploded.select(col("article.text").cast("string").alias("text"))
        out = out.withColumn("origin", lit(src_prefix))
        return out
    except Exception as e:
        print(f"⚠️ extract_from_articles failed: {e}")
        return None

def extract_text_generic(df, src_prefix):
    # Preference: if 'text' in columns -> use it
    # elif 'content' in columns -> use content
    # elif 'articles' -> explode articles -> use article.text
    if df is None:
        return None
    cols = df.columns
    if "text" in cols:
        return df.select(col("text").cast("string").alias("text")).withColumn("origin", lit(src_prefix))
    if "content" in cols:
        return df.select(col("content").cast("string").alias("text")).withColumn("origin", lit(src_prefix))
    if "articles" in cols:
        return extract_from_articles(df, src_prefix)
    print(f"⚠️ No 'text'/'content'/'articles' found in columns {cols} for source {src_prefix}")
    return None

def build_id_expr():
    # deterministic id: md5("src_<origin>_<text>")
    return md5(concat(lit("src_"), col("origin"), lit("_"), col("text")))

# -------------------
# Main
# -------------------
def main(args):
    spark = build_spark(args.minio_endpoint, args.access_key, args.secret_key)
    raw_base = f"s3a://{args.bucket}/{args.raw_prefix}".rstrip("/")
    out_base = f"s3a://{args.bucket}/{args.out_prefix}".rstrip("/")

    sources = []

    # corpus.csv -> 'text'
    print("Reading corpus.csv ...")
    df = read_csv_text(spark, raw_base + "/corpus.csv", "text")
    if df is not None:
        df = df.withColumn("origin", lit("corpus.csv"))
        sources.append(df)

    # data.csv -> 'full_text' -> rename to text
    print("Reading data.csv ...")
    df2 = read_csv_text(spark, raw_base + "/data.csv", "full_text")
    if df2 is not None:
        df2 = df2.select(col("text").alias("text")).withColumn("origin", lit("data.csv"))
        sources.append(df2)

    # updated_legal_corpus.csv -> 'content'
    print("Reading updated_legal_corpus.csv ...")
    df3 = read_csv_text(spark, raw_base + "/updated_legal_corpus.csv", "content")
    if df3 is not None:
        df3 = df3.withColumn("origin", lit("updated_legal_corpus.csv"))
        sources.append(df3)

    # legal_corpus.json -> nested articles[].text
    print("Reading legal_corpus.json ...")
    legal_raw = safe_read_json(spark, raw_base + "/legal_corpus.json", multiline=True)
    legal_text = extract_text_generic(legal_raw, "legal_corpus.json")
    if legal_text is not None:
        legal_text = legal_text.withColumn("origin", lit("legal_corpus.json"))
        sources.append(legal_text)

    # zalo_corpus.json -> nested articles[].text
    print("Reading zalo_corpus.json ...")
    zalo_raw = safe_read_json(spark, raw_base + "/zalo_corpus.json", multiline=True)
    zalo_text = extract_text_generic(zalo_raw, "zalo_corpus.json")
    if zalo_text is not None:
        zalo_text = zalo_text.withColumn("origin", lit("zalo_corpus.json"))
        sources.append(zalo_text)

    # vbpl_crawl.json -> prefers 'text' else 'content'
    print("Reading vbpl_crawl.json ...")
    vbpl_raw = safe_read_json(spark, raw_base + "/vbpl_crawl.json", multiline=True)
    vbpl_text = extract_text_generic(vbpl_raw, "vbpl_crawl.json")
    if vbpl_text is not None:
        vbpl_text = vbpl_text.withColumn("origin", lit("vbpl_crawl.json"))
        sources.append(vbpl_text)

    valid = [df for df in sources if df is not None]
    if len(valid) == 0:
        print("❌ No valid sources read. Exiting.")
        spark.stop()
        sys.exit(1)

    df_union = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), valid)

    # clean text
    df_clean = df_union.withColumn("text", trim(col("text"))) \
                       .filter((col("text").isNotNull()) & (col("text") != ""))

    # add deterministic id
    df_final = df_clean.withColumn("id", build_id_expr()).select("id", "text")

    # deduplicate by id
    df_final = df_final.dropDuplicates(["id"])

    # convert to JSON lines RDD
    json_rdd = df_final.toJSON()

    tmp_out = out_base + "/jsonl_tmp"
    final_object_path = out_base + "/combined.jsonl"

    # delete tmp_out if exists (best-effort)
    try:
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        path_to_delete = spark._jvm.org.apache.hadoop.fs.Path(tmp_out)
        if hadoop_fs.exists(path_to_delete):
            hadoop_fs.delete(path_to_delete, True)
    except Exception:
        pass

    # coalesce if requested
    if args.coalesce:
        print("Coalescing to 1 partition (may be heavy for large datasets)...")
        json_rdd = json_rdd.coalesce(1)

    json_rdd.saveAsTextFile(tmp_out)
    print(f"Wrote temporary jsonl parts to {tmp_out}")

    # try to rename single part -> combined.jsonl (works if coalesced to 1 part or at least one part exists)
    # --- replace move/rename logic with FS for s3a URI ---
    try:
        conf = spark._jsc.hadoopConfiguration()
        URI = spark._jvm.java.net.URI
        Path = spark._jvm.org.apache.hadoop.fs.Path
        FileSystem = spark._jvm.org.apache.hadoop.fs.FileSystem

        src_path = Path(tmp_out)
        dest_path = Path(final_object_path)

        # get FS for the src (s3a) using its URI
        fs_src = FileSystem.get(URI(src_path.toString()), conf)
        # ensure dest parent exists (optional)
        if fs_src.exists(dest_path):
            fs_src.delete(dest_path, True)

        # list part file under tmp_out (on the same FS)
        statuses = fs_src.listStatus(src_path)
        part = None
        for status in statuses:
            name = status.getPath().getName()
            if name.startswith("part-"):
                part = status.getPath()
                break

        if part is None:
            print("⚠️ No part file found under tmp output; leaving parts in place.")
        else:
            # rename the part -> final path (same FS)
            fs_src.rename(part, dest_path)
            # remove tmp dir
            fs_src.delete(src_path, True)
            print(f"✅ combined JSONL available at: {final_object_path}")

    except Exception as e:
        print(f"⚠️ Could not rename/move part file to combined.jsonl automatically: {e}")
        print(f"JSON parts remain at {tmp_out}. Use mc/aws-cli to merge if needed.")


    spark.stop()
    print("All done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--minio-endpoint", default="http://localhost:9000")
    p.add_argument("--access-key", default="minio_access_key")
    p.add_argument("--secret-key", default="minio_secret_key")
    p.add_argument("--bucket", default="datalake")
    p.add_argument("--raw-prefix", default="raw/rag_corpus")
    p.add_argument("--out-prefix", default="processed/rag_corpus")
    p.add_argument("--coalesce", action="store_true", help="Coalesce to single file (may be heavy)")
    args = p.parse_args()
    main(args)
