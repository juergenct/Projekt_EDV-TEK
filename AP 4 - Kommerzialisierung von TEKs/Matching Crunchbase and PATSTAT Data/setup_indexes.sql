-- PATSTAT Database Index Setup

BEGIN;

-- Essential indexes for person table
CREATE INDEX IF NOT EXISTS idx_tls206_person_name_lower 
ON tls206_person(LOWER(person_name));

CREATE INDEX IF NOT EXISTS idx_tls206_person_ctry_code 
ON tls206_person(person_ctry_code);

CREATE INDEX IF NOT EXISTS idx_tls206_person_sector 
ON tls206_person(psn_sector) 
WHERE psn_sector NOT IN ('INDIVIDUAL', 'UNKNOWN');

CREATE INDEX IF NOT EXISTS idx_tls206_person_id 
ON tls206_person(person_id);

-- Indexes for person-application junction table
CREATE INDEX IF NOT EXISTS idx_tls207_pers_appln_person_id 
ON tls207_pers_appln(person_id);

CREATE INDEX IF NOT EXISTS idx_tls207_pers_appln_appln_id 
ON tls207_pers_appln(appln_id);

CREATE INDEX IF NOT EXISTS idx_tls207_pers_appln_composite 
ON tls207_pers_appln(person_id, appln_id, applt_seq_nr, invt_seq_nr);

-- Indexes for application table
CREATE INDEX IF NOT EXISTS idx_tls201_appln_auth_year 
ON tls201_appln(appln_auth, CAST(appln_filing_year AS INTEGER)) 
WHERE appln_auth IN ('EP', 'US');

CREATE INDEX IF NOT EXISTS idx_tls201_appln_id 
ON tls201_appln(CAST(appln_id AS INTEGER));

-- Index for family information (used in country code refilling)
CREATE INDEX IF NOT EXISTS idx_tls201_appln_docdb_family 
ON tls201_appln(CAST(docdb_family_id AS INTEGER));

-- Additional indexes for sequence numbers (all stored as text)
CREATE INDEX IF NOT EXISTS idx_tls207_pers_appln_applt_seq 
ON tls207_pers_appln(CAST(applt_seq_nr AS INTEGER)) 
WHERE applt_seq_nr IS NOT NULL AND applt_seq_nr != '';

CREATE INDEX IF NOT EXISTS idx_tls207_pers_appln_invt_seq 
ON tls207_pers_appln(CAST(invt_seq_nr AS INTEGER)) 
WHERE invt_seq_nr IS NOT NULL AND invt_seq_nr != '';

-- Create a materialized view for faster company queries (optional but recommended)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ip5_applicants AS
SELECT 
    CAST(p.person_id AS INTEGER) as person_id,
    p.person_name,
    p.person_ctry_code,
    p.psn_sector,
    COUNT(DISTINCT CAST(a.appln_id AS INTEGER)) as patent_count
FROM tls206_person p
JOIN tls207_pers_appln pa ON CAST(p.person_id AS INTEGER) = CAST(pa.person_id AS INTEGER)
JOIN tls201_appln a ON CAST(pa.appln_id AS INTEGER) = CAST(a.appln_id AS INTEGER)
WHERE 
    a.appln_auth IN ('EP', 'US')
    AND CAST(a.appln_filing_year AS INTEGER) >= 2000
    AND CAST(pa.applt_seq_nr AS INTEGER) > 0
    AND p.psn_sector NOT IN ('INDIVIDUAL', 'UNKNOWN')
GROUP BY CAST(p.person_id AS INTEGER), p.person_name, p.person_ctry_code, p.psn_sector;

CREATE INDEX IF NOT EXISTS idx_mv_ip5_applicants_country ON mv_ip5_applicants(person_ctry_code);
CREATE INDEX IF NOT EXISTS idx_mv_ip5_applicants_name ON mv_ip5_applicants(LOWER(person_name));

-- Update table statistics for query planner
ANALYZE tls206_person;
ANALYZE tls207_pers_appln;
ANALYZE tls201_appln;

COMMIT;

-- Verify indexes were created (simplified query)
SELECT 
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE schemaname = 'public'
AND tablename IN ('tls206_person', 'tls207_pers_appln', 'tls201_appln', 'mv_ip5_applicants')
ORDER BY tablename, indexname;