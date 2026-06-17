-- =====================================================================
-- retail_forecasting / sql / load_data.sql
-- =====================================================================
-- Postgres COPY commands to load the processed CSVs into the schema.
-- Run schema.sql first.
--
-- Adjust the file paths to match your local setup (or use \copy from psql
-- to avoid the server-side path requirement).
-- =====================================================================


-- dim_store from store.csv
\copy dim_store(store_id, store_type, assortment, competition_distance,
                competition_open_year, promo2, promo2_since_year, promo_interval)
FROM 'data/raw/store_clean.csv' DELIMITER ',' CSV HEADER;


-- dim_date — generate it instead of loading
INSERT INTO dim_date
SELECT
    d::date                                                AS date_key,
    EXTRACT(YEAR    FROM d)::int                           AS year,
    EXTRACT(QUARTER FROM d)::int                           AS quarter,
    EXTRACT(MONTH   FROM d)::int                           AS month,
    TRIM(TO_CHAR(d, 'Month'))                              AS month_name,
    EXTRACT(WEEK    FROM d)::int                           AS week_of_year,
    EXTRACT(DAY     FROM d)::int                           AS day_of_month,
    EXTRACT(ISODOW  FROM d)::int - 1                       AS day_of_week,
    TRIM(TO_CHAR(d, 'Day'))                                AS day_name,
    EXTRACT(ISODOW FROM d) IN (6, 7)                       AS is_weekend,
    d = DATE_TRUNC('month', d)                             AS is_month_start,
    d = (DATE_TRUNC('month', d) + INTERVAL '1 month - 1 day')::date AS is_month_end
FROM generate_series('2013-01-01'::date, '2015-09-30'::date, '1 day'::interval) d;


-- fact_sales from train.csv (filter Open=1, Sales>0 first in your prep step)
\copy fact_sales(store_id, date_key, sales, customers, promo, school_holiday, state_holiday)
FROM 'data/processed/fact_sales.csv' DELIMITER ',' CSV HEADER;


-- fact_forecast from predictions_long.csv (after pivoting to one row per model)
\copy fact_forecast(store_id, date_key, model_name, predicted_sales)
FROM 'data/processed/fact_forecast.csv' DELIMITER ',' CSV HEADER;


-- quick row count check
SELECT 'dim_store'    AS tbl, COUNT(*) FROM dim_store
UNION ALL SELECT 'dim_date',     COUNT(*) FROM dim_date
UNION ALL SELECT 'fact_sales',   COUNT(*) FROM fact_sales
UNION ALL SELECT 'fact_forecast',COUNT(*) FROM fact_forecast;
