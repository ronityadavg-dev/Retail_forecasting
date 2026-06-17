-- =====================================================================
-- retail_forecasting / sql / schema.sql
-- =====================================================================
-- Star schema for the Rossmann sales data. Designed for Tableau:
-- one fact table (daily sales) and supporting dim tables.
--
-- Built for PostgreSQL but the syntax is portable enough for MySQL or
-- SQLite with minor tweaks (mainly the SERIAL keyword).
-- =====================================================================

DROP TABLE IF EXISTS fact_sales       CASCADE;
DROP TABLE IF EXISTS dim_store        CASCADE;
DROP TABLE IF EXISTS dim_date         CASCADE;
DROP TABLE IF EXISTS fact_forecast    CASCADE;


-- ---------------------------------------------------------------------
-- dim_store — one row per store, holds all the metadata
-- ---------------------------------------------------------------------
CREATE TABLE dim_store (
    store_id              INTEGER PRIMARY KEY,
    store_type            CHAR(1),
    assortment            CHAR(1),
    competition_distance  NUMERIC(10, 2),
    competition_open_year INTEGER,
    promo2                BOOLEAN,
    promo2_since_year     INTEGER,
    promo_interval        VARCHAR(40)
);


-- ---------------------------------------------------------------------
-- dim_date — standard calendar dim. Build it from the date range.
-- ---------------------------------------------------------------------
CREATE TABLE dim_date (
    date_key        DATE PRIMARY KEY,
    year            INTEGER,
    quarter         INTEGER,
    month           INTEGER,
    month_name      VARCHAR(12),
    week_of_year    INTEGER,
    day_of_month    INTEGER,
    day_of_week     INTEGER,        -- 0=Mon .. 6=Sun
    day_name        VARCHAR(10),
    is_weekend      BOOLEAN,
    is_month_start  BOOLEAN,
    is_month_end    BOOLEAN
);


-- ---------------------------------------------------------------------
-- fact_sales — grain = one row per store per day (open days only)
-- ---------------------------------------------------------------------
CREATE TABLE fact_sales (
    sale_id         BIGSERIAL PRIMARY KEY,
    store_id        INTEGER NOT NULL REFERENCES dim_store(store_id),
    date_key        DATE    NOT NULL REFERENCES dim_date(date_key),
    sales           NUMERIC(12, 2),
    customers       INTEGER,
    promo           BOOLEAN,
    school_holiday  BOOLEAN,
    state_holiday   CHAR(1)         -- '0','a','b','c'
);

CREATE INDEX idx_fact_sales_store ON fact_sales(store_id);
CREATE INDEX idx_fact_sales_date  ON fact_sales(date_key);


-- ---------------------------------------------------------------------
-- fact_forecast — predictions from each model, one row per
--                 store/date/model combination
-- ---------------------------------------------------------------------
CREATE TABLE fact_forecast (
    forecast_id     BIGSERIAL PRIMARY KEY,
    store_id        INTEGER NOT NULL REFERENCES dim_store(store_id),
    date_key        DATE    NOT NULL REFERENCES dim_date(date_key),
    model_name      VARCHAR(20) NOT NULL,    -- 'SARIMA' | 'Prophet' | 'XGBoost'
    predicted_sales NUMERIC(12, 2)
);

CREATE INDEX idx_fact_forecast_store ON fact_forecast(store_id);
CREATE INDEX idx_fact_forecast_date  ON fact_forecast(date_key);
CREATE INDEX idx_fact_forecast_model ON fact_forecast(model_name);
