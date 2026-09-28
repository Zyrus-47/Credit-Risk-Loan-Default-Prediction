-- Portfolio aggregation views and risk benchmarks

CREATE TABLE IF NOT EXISTS grade_risk_summary AS
SELECT
    grade,
    COUNT(*) AS total_loans,
    SUM(default_flag) AS total_defaults,
    ROUND(AVG(default_flag) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate,
    ROUND(AVG(dti), 2) AS avg_dti,
    ROUND(AVG(annual_inc), 2) AS avg_annual_inc,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amnt,
    ROUND(SUM(loan_amnt), 2) AS total_funded_volume
FROM cleaned_loans
GROUP BY grade
ORDER BY grade;

CREATE TABLE IF NOT EXISTS purpose_risk_summary AS
SELECT
    purpose,
    term,
    COUNT(*) AS loan_count,
    ROUND(AVG(default_flag) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate,
    ROUND(AVG(revol_util), 2) AS avg_revol_util
FROM cleaned_loans
GROUP BY purpose, term
ORDER BY loan_count DESC;

CREATE TABLE IF NOT EXISTS delinquency_cohort_summary AS
SELECT
    CASE 
        WHEN delinq_2yrs = 0 THEN '0 Delinquencies'
        WHEN delinq_2yrs = 1 THEN '1 Delinquency'
        ELSE '2+ Delinquencies'
    END AS delinq_bucket,
    CASE
        WHEN inq_last_6mths = 0 THEN '0 Inquiries'
        WHEN inq_last_6mths BETWEEN 1 AND 2 THEN '1-2 Inquiries'
        ELSE '3+ Inquiries'
    END AS inq_bucket,
    COUNT(*) AS loan_count,
    ROUND(AVG(default_flag) * 100.0, 2) AS default_rate_pct
FROM cleaned_loans
GROUP BY 1, 2
ORDER BY 1, 2;
