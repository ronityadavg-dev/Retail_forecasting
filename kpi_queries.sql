-- =====================================================================
-- retail_forecasting / sql / kpi_queries.sql
-- =====================================================================
-- Queries powering the Tableau dashboard. Each one is named so it's
-- easy to map back to a specific worksheet.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Q1: headline KPIs — total sales, total customers, avg basket, YoY %
-- ---------------------------------------------------------------------
WITH this_year AS (
    SELECT
        SUM(sales)                                        AS total_sales,
        SUM(customers)                                    AS total_customers,
        SUM(sales) * 1.0 / NULLIF(SUM(customers), 0)      AS avg_basket
    FROM fact_sales fs
    JOIN dim_date d ON fs.date_key = d.date_key
    WHERE d.year = 2015
),
last_year AS (
    SELECT SUM(sales) AS total_sales
    FROM fact_sales fs
    JOIN dim_date d ON fs.date_key = d.date_key
    WHERE d.year = 2014
)
SELECT
    ty.total_sales,
    ty.total_customers,
    ROUND(ty.avg_basket::numeric, 2)                           AS avg_basket,
    ROUND(((ty.total_sales - ly.total_sales) / ly.total_sales * 100)::numeric, 2) AS yoy_growth_pct
FROM this_year ty CROSS JOIN last_year ly;


-- ---------------------------------------------------------------------
-- Q2: monthly sales trend with promo flag rate
-- ---------------------------------------------------------------------
SELECT
    d.year,
    d.month,
    d.month_name,
    SUM(fs.sales)                                              AS monthly_sales,
    AVG(CASE WHEN fs.promo THEN 1.0 ELSE 0.0 END)              AS promo_day_rate,
    COUNT(DISTINCT fs.store_id)                                AS active_stores
FROM fact_sales fs
JOIN dim_date d ON fs.date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- ---------------------------------------------------------------------
-- Q3: top + bottom 10 stores by total sales
-- ---------------------------------------------------------------------
WITH ranked AS (
    SELECT
        fs.store_id,
        ds.store_type,
        ds.assortment,
        SUM(fs.sales) AS total_sales,
        RANK() OVER (ORDER BY SUM(fs.sales) DESC) AS rk_top,
        RANK() OVER (ORDER BY SUM(fs.sales) ASC)  AS rk_bot
    FROM fact_sales fs
    JOIN dim_store ds ON fs.store_id = ds.store_id
    GROUP BY fs.store_id, ds.store_type, ds.assortment
)
SELECT * FROM ranked WHERE rk_top <= 10 OR rk_bot <= 10
ORDER BY total_sales DESC;


-- ---------------------------------------------------------------------
-- Q4: promo lift — avg sales with vs without promo, per store type
-- ---------------------------------------------------------------------
SELECT
    ds.store_type,
    AVG(CASE WHEN fs.promo     THEN fs.sales END) AS avg_sales_promo,
    AVG(CASE WHEN NOT fs.promo THEN fs.sales END) AS avg_sales_no_promo,
    ROUND((
        AVG(CASE WHEN fs.promo     THEN fs.sales END) /
        NULLIF(AVG(CASE WHEN NOT fs.promo THEN fs.sales END), 0)
        - 1
    )::numeric * 100, 2) AS promo_lift_pct
FROM fact_sales fs
JOIN dim_store ds ON fs.store_id = ds.store_id
GROUP BY ds.store_type
ORDER BY ds.store_type;


-- ---------------------------------------------------------------------
-- Q5: day-of-week heatmap data (store_type x day_of_week)
-- ---------------------------------------------------------------------
SELECT
    ds.store_type,
    d.day_name,
    d.day_of_week,
    AVG(fs.sales) AS avg_sales
FROM fact_sales fs
JOIN dim_store ds ON fs.store_id = ds.store_id
JOIN dim_date  d  ON fs.date_key = d.date_key
GROUP BY ds.store_type, d.day_name, d.day_of_week
ORDER BY ds.store_type, d.day_of_week;


-- ---------------------------------------------------------------------
-- Q6: actual vs forecast for the holdout window — feeds the
--     comparison view in the dashboard
-- ---------------------------------------------------------------------
SELECT
    fs.date_key,
    fs.store_id,
    fs.sales        AS actual,
    ff.model_name,
    ff.predicted_sales,
    (ff.predicted_sales - fs.sales)                                       AS residual,
    ABS(ff.predicted_sales - fs.sales) / NULLIF(fs.sales, 0) * 100        AS abs_pct_error
FROM fact_sales fs
JOIN fact_forecast ff
  ON fs.store_id = ff.store_id
 AND fs.date_key = ff.date_key
WHERE fs.date_key > (SELECT MAX(date_key) - INTERVAL '42 days' FROM fact_sales)
ORDER BY fs.store_id, fs.date_key, ff.model_name;


-- ---------------------------------------------------------------------
-- Q7: model accuracy summary
-- ---------------------------------------------------------------------
WITH pairs AS (
    SELECT
        ff.model_name,
        fs.sales            AS actual,
        ff.predicted_sales  AS predicted
    FROM fact_sales fs
    JOIN fact_forecast ff
      ON fs.store_id = ff.store_id AND fs.date_key = ff.date_key
)
SELECT
    model_name,
    COUNT(*)                                                  AS n_obs,
    SQRT(AVG(POWER(predicted - actual, 2)))                   AS rmse,
    AVG(ABS(predicted - actual) / NULLIF(actual, 0)) * 100    AS mape_pct,
    AVG(predicted - actual)                                   AS bias
FROM pairs
GROUP BY model_name
ORDER BY rmse;
