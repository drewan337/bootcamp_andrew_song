# Data Storage Pipeline

## Folder Structure & File Formats
| Format  | Folder      | Purpose                                                                 |
|---------|------------|-------------------------------------------------------------------------|
| CSV     | `data/raw/` | Raw data storage - Human-readable, interoperable with all tools         |
| Parquet | `data/processed/` | Processed data - Faster I/O, preserves data types, columnar storage |

## Environment Configuration
The pipeline uses `.env` for path management
The code uses environment variables from .env (or defaults) to configure storage paths dynamically.
It loads DATA_DIR_RAW and DATA_DIR_PROCESSED via python-dotenv, creates missing directories automatically, and saves/reads files (CSV to raw/, Parquet to processed/) using these paths.