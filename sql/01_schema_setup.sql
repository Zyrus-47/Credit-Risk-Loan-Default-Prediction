-- 01_schema_setup.sql
-- Table DDL and indexes for raw and staging LendingClub loan data
-- Compatible with DuckDB, SQLite, and PostgreSQL dialects

CREATE TABLE IF NOT EXISTS raw_lending_club_loans (
    id VARCHAR(64) PRIMARY KEY,
    loan_amnt DOUBLE,
    funded_amnt DOUBLE,
    term VARCHAR(32),
    int_rate VARCHAR(32),
    installment DOUBLE,
    grade VARCHAR(8),
    sub_grade VARCHAR(8),
    emp_title VARCHAR(255),
    emp_length VARCHAR(32),
    home_ownership VARCHAR(32),
    annual_inc DOUBLE,
    verification_status VARCHAR(64),
    issue_d VARCHAR(32),
    loan_status VARCHAR(128),
    purpose VARCHAR(64),
    title VARCHAR(255),
    dti DOUBLE,
    delinq_2yrs DOUBLE,
    earliest_cr_line VARCHAR(32),
    inq_last_6mths DOUBLE,
    open_acc DOUBLE,
    pub_rec DOUBLE,
    revol_bal DOUBLE,
    revol_util VARCHAR(32),
    total_acc DOUBLE,
    total_rec_prncp DOUBLE,
    recoveries DOUBLE,
    collection_recovery_fee DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_raw_loan_status ON raw_lending_club_loans(loan_status);
CREATE INDEX IF NOT EXISTS idx_raw_grade ON raw_lending_club_loans(grade);
