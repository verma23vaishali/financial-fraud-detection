-- Run against a table named creditcard in DuckDB/SQLite.

SELECT COUNT(*) AS total_transactions,
       SUM(CASE WHEN Class=1 THEN 1 ELSE 0 END) AS fraud_transactions,
       ROUND(100.0 * SUM(CASE WHEN Class=1 THEN 1 ELSE 0 END)/COUNT(*),4) AS fraud_rate_pct
FROM creditcard;

SELECT Class, COUNT(*) AS transaction_count,
       ROUND(AVG(Amount),2) AS avg_amount,
       ROUND(MAX(Amount),2) AS max_amount
FROM creditcard GROUP BY Class;

SELECT
  CASE WHEN Amount < 25 THEN '<25'
       WHEN Amount < 100 THEN '25-99'
       WHEN Amount < 500 THEN '100-499'
       WHEN Amount < 1000 THEN '500-999'
       ELSE '1000+' END AS amount_band,
  COUNT(*) AS transactions,
  SUM(Class) AS fraud_count
FROM creditcard
GROUP BY amount_band
ORDER BY fraud_count DESC;

SELECT CAST((Time/3600)%24 AS INTEGER) AS hour,
       COUNT(*) AS transactions, SUM(Class) AS fraud_count
FROM creditcard
GROUP BY hour ORDER BY hour;
