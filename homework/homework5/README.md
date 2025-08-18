# Data Storage Pipeline

## Folder Structure
## File Formats
| Format  | Folder      | Purpose                                                                 |
|---------|------------|-------------------------------------------------------------------------|
| CSV     | `data/raw/` | Raw data storage - Human-readable, interoperable with all tools         |
| Parquet | `data/processed/` | Processed data - Faster I/O, preserves data types, columnar storage |

## Environment Configuration
The pipeline uses `.env` for path management