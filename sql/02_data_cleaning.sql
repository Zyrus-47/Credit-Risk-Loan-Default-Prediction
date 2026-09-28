-- 02_data_cleaning.sql
-- Core data hygiene, sanitization, string parsing, and target variable definition
-- Filters out active/ongoing loans to eliminate survivorship bias and target leakage

CREATE TABLE IF NOT EXISTS cleaned_loans AS
SELECT
    id,
    CAST(loan_amnt AS DOUBLE) AS loan_amnt,
    CAST(funded_amnt AS DOUBLE) AS funded_amnt,
    
    -- Clean parsing: Strip ' months' from term and cast to integer
    CAST(TRIM(REPLACE(term, ' months', '')) AS INTEGER) AS term,
    
    -- Clean percentage symbol and whitespace from int_rate
    CAST(TRIM(REPLACE(REPLACE(CAST(int_rate AS VARCHAR), '%', ''), ' ', '')) AS DOUBLE) AS int_rate,
    
    CAST(installment AS DOUBLE) AS installment,
    TRIM(grade) AS grade,
    TRIM(sub_grade) AS sub_grade,
    
    -- Standardize emp_length: '< 1 year' -> 0, '10+ years' -> 10, intermediate integer years
    CASE
        WHEN emp_length LIKE '%< 1%' THEN 0
        WHEN emp_length LIKE '%10+%' THEN 10
        WHEN emp_length LIKE '%1 year%' THEN 1
        WHEN emp_length LIKE '%2 year%' THEN 2
        WHEN emp_length LIKE '%3 year%' THEN 3
        WHEN emp_length LIKE '%4 year%' THEN 4
        WHEN emp_length LIKE '%5 year%' THEN 5
        WHEN emp_length LIKE '%6 year%' THEN 6
        WHEN emp_length LIKE '%7 year%' THEN 7
        WHEN emp_length LIKE '%8 year%' THEN 8
        WHEN emp_length LIKE '%9 year%' THEN 9
        WHEN emp_length IS NULL OR emp_length = 'n/a' OR emp_length = '' THEN NULL
        ELSE 0
    END AS emp_length,
    
    TRIM(home_ownership) AS home_ownership,
    CAST(annual_inc AS DOUBLE) AS annual_inc,
    TRIM(verification_status) AS verification_status,
    issue_d,
    loan_status,
    TRIM(purpose) AS purpose,
    CAST(dti AS DOUBLE) AS dti,
    CAST(delinq_2yrs AS DOUBLE) AS delinq_2yrs,
    earliest_cr_line,
    CAST(inq_last_6mths AS DOUBLE) AS inq_last_6mths,
    CAST(open_acc AS DOUBLE) AS open_acc,
    CAST(pub_rec AS DOUBLE) AS pub_rec,
    CAST(revol_bal AS DOUBLE) AS revol_bal,
    
    -- Clean percentage symbol and whitespace from revol_util
    CAST(TRIM(REPLACE(REPLACE(CAST(revol_util AS VARCHAR), '%', ''), ' ', '')) AS DOUBLE) AS revol_util,
    
    CAST(total_acc AS DOUBLE) AS total_acc,
    COALESCE(CAST(recoveries AS DOUBLE), 0.0) AS recoveries,
    COALESCE(CAST(collection_recovery_fee AS DOUBLE), 0.0) AS collection_recovery_fee,
    COALESCE(CAST(total_rec_prncp AS DOUBLE), 0.0) AS total_rec_prncp,
    
    -- Target Definition:
    -- 1 (Default / Charge-Off)
    -- 0 (Fully Paid)
    CASE 
        WHEN loan_status IN (
            'Charged Off',
            'Default',
            'Does not meet the credit policy. Status:Charged Off'
        ) THEN 1
        WHEN loan_status IN (
            'Fully Paid',
            'Does not meet the credit policy. Status:Fully Paid'
        ) THEN 0
        ELSE NULL
    END AS default_flag

FROM raw_lending_club_loans
WHERE 
    -- Eliminate ongoing/active loans to prevent survivorship bias and target leakage
    loan_status NOT IN ('Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)')
    AND loan_status IS NOT NULL
    AND (
        loan_status IN ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off')
        OR loan_status IN ('Fully Paid', 'Does not meet the credit policy. Status:Fully Paid')
    );
