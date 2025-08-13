import pandas as pd
import numpy as np
import psycopg2
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import jellyfish
import Levenshtein
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str


class NameHarmonizer:
    """Harmonizes company and person names for matching"""
    
    # Legal designations to remove
    LEGAL_TERMS = [
        r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|llp|gmbh|ag|sa|srl|spa|plc|pvt|pte|pty|co|company)\b',
        r'\b(holdings|group|partners|associates|ventures|capital|technologies|tech|solutions|systems|services)\b',
    ]
    
    # Academic titles for person names
    TITLES = [
        r'\b(dr|prof|professor|phd|md|msc|bsc|ma|ba|ing|ir|mr|mrs|ms|miss)\b',
    ]
    
    @staticmethod
    def harmonize_company(name: str) -> str:
        """Harmonize company names"""
        if pd.isna(name) or not name:
            return ""
        
        name = name.lower().strip()
        
        # Remove legal designations
        for pattern in NameHarmonizer.LEGAL_TERMS:
            name = re.sub(pattern, ' ', name)
        
        # Remove punctuation and normalize spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = ' '.join(name.split())
        
        return name
    
    @staticmethod
    def harmonize_person(name: str) -> str:
        """Harmonize person names"""
        if pd.isna(name) or not name:
            return ""
        
        name = name.lower().strip()
        
        # Remove titles
        for pattern in NameHarmonizer.TITLES:
            name = re.sub(pattern, ' ', name)
        
        # Remove c/o company names
        name = re.sub(r'\bc/o\b.*', '', name)
        
        # Basic ASCII conversion and cleanup
        name = re.sub(r'[^\w\s]', ' ', name)
        name = ' '.join(name.split())
        
        return name


class Matcher:
    """String matching algorithms"""
    
    @staticmethod
    def calculate_similarity(str1: str, str2: str, method: str = 'combined') -> float:
        """Calculate string similarity using specified method"""
        if not str1 or not str2:
            return 0.0
        
        if method == 'perfect':
            return 1.0 if str1 == str2 else 0.0
        
        elif method == 'jaro_winkler':
            return jellyfish.jaro_winkler_similarity(str1, str2)
        
        elif method == 'ngram':
            # 2-gram similarity as emphasized in paper
            def get_ngrams(text: str, n: int = 2):
                return set(text[i:i+n] for i in range(len(text) - n + 1))
            
            if len(str1) < 2 or len(str2) < 2:
                return 0.0
            
            ng1 = get_ngrams(str1)
            ng2 = get_ngrams(str2)
            
            if not ng1 or not ng2:
                return 0.0
            
            return len(ng1 & ng2) / len(ng1 | ng2)
        
        elif method == 'combined':
            # Weighted combination of methods
            scores = {
                'perfect': 1.0 if str1 == str2 else 0.0,
                'jaro': jellyfish.jaro_winkler_similarity(str1, str2),
                'ngram': Matcher.calculate_similarity(str1, str2, 'ngram')
            }
            
            # Weights from paper methodology
            return (scores['perfect'] * 0.5 + 
                   scores['jaro'] * 0.3 + 
                   scores['ngram'] * 0.2)
        
        return 0.0


class PATSTATConnector:
    """Handles PATSTAT database queries"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        self.conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
        logger.info("Connected to PATSTAT database")
    
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from PATSTAT")
    
    def get_applicants(self, batch_size: int = 100000) -> pd.DataFrame:
        """Get IP5 applicants in batches"""
        query = """
        SELECT DISTINCT
            person_id,
            person_name,
            person_ctry_code,
            psn_sector
        FROM tls206_person
        WHERE person_id IN (
            SELECT DISTINCT pa.person_id
            FROM tls207_pers_appln pa
            JOIN tls201_appln a ON pa.appln_id = a.appln_id
            WHERE a.appln_auth IN ('EP', 'US')
            AND a.appln_filing_year >= 2000
            AND pa.applt_seq_nr > 0
        )
        AND psn_sector NOT IN ('INDIVIDUAL', 'UNKNOWN')
        LIMIT %s OFFSET %s
        """
        
        all_data = []
        offset = 0
        
        with self.conn.cursor() as cur:
            while True:
                cur.execute(query, (batch_size, offset))
                batch = cur.fetchall()
                if not batch:
                    break
                
                df = pd.DataFrame(batch, columns=['person_id', 'person_name', 'person_ctry_code', 'psn_sector'])
                all_data.append(df)
                offset += batch_size
                logger.info(f"Loaded {offset} applicants...")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def get_inventors(self, batch_size: int = 100000) -> pd.DataFrame:
        """Get IP5 inventors in batches"""
        query = """
        SELECT DISTINCT
            person_id,
            person_name,
            person_ctry_code
        FROM tls206_person
        WHERE person_id IN (
            SELECT DISTINCT pa.person_id
            FROM tls207_pers_appln pa
            JOIN tls201_appln a ON pa.appln_id = a.appln_id
            WHERE a.appln_auth IN ('EP', 'US')
            AND a.appln_filing_year >= 1978
            AND pa.invt_seq_nr > 0
        )
        LIMIT %s OFFSET %s
        """
        
        all_data = []
        offset = 0
        
        with self.conn.cursor() as cur:
            while True:
                cur.execute(query, (batch_size, offset))
                batch = cur.fetchall()
                if not batch:
                    break
                
                df = pd.DataFrame(batch, columns=['person_id', 'person_name', 'person_ctry_code'])
                all_data.append(df)
                offset += batch_size
                logger.info(f"Loaded {offset} inventors...")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def match_companies_batch(cb_batch: pd.DataFrame, ps_data: pd.DataFrame, threshold: float = 0.85) -> List[Dict]:
    """Match a batch of companies"""
    matches = []
    harmonizer = NameHarmonizer()
    matcher = Matcher()
    
    for _, cb_row in cb_batch.iterrows():
        cb_name = cb_row.get('harmonized_name', '')
        cb_country = cb_row.get('country_code', '')
        
        if not cb_name:
            continue
        
        # Filter by country if available
        if pd.notna(cb_country) and cb_country:
            candidates = ps_data[ps_data['person_ctry_code'] == cb_country]
        else:
            candidates = ps_data
        
        if candidates.empty:
            continue
        
        # Find best match
        best_score = 0
        best_match = None
        
        for _, ps_row in candidates.iterrows():
            ps_name = ps_row.get('harmonized_name', '')
            if not ps_name:
                continue
            
            score = matcher.calculate_similarity(cb_name, ps_name, 'combined')
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = {
                    'cb_uuid': cb_row['uuid'],
                    'cb_name': cb_row['name'],
                    'ps_person_id': ps_row['person_id'],
                    'ps_name': ps_row['person_name'],
                    'country_code': cb_country,
                    'match_score': score
                }
        
        if best_match:
            matches.append(best_match)
    
    return matches


def match_people_batch(cb_batch: pd.DataFrame, ps_data: pd.DataFrame, threshold: float = 0.75) -> List[Dict]:
    """Match a batch of people using 2-gram similarity"""
    matches = []
    matcher = Matcher()
    
    for _, cb_row in cb_batch.iterrows():
        cb_name = cb_row.get('harmonized_name', '')
        if not cb_name or len(cb_name) < 3:
            continue
        
        # Create blocking on first character of last name
        cb_tokens = cb_name.split()
        if not cb_tokens:
            continue
        
        first_char = cb_tokens[-1][0] if cb_tokens else ''
        candidates = ps_data[ps_data['harmonized_name'].str.contains(first_char, na=False)]
        
        if candidates.empty:
            continue
        
        # Find best match using 2-gram similarity
        best_score = 0
        best_match = None
        
        for _, ps_row in candidates.iterrows():
            ps_name = ps_row.get('harmonized_name', '')
            if not ps_name:
                continue
            
            # Use ngram similarity as emphasized in paper
            score = matcher.calculate_similarity(cb_name, ps_name, 'ngram')
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = {
                    'cb_uuid': cb_row['uuid'],
                    'cb_name': cb_row.get('name', ''),
                    'ps_person_id': ps_row['person_id'],
                    'ps_name': ps_row['person_name'],
                    'match_score': score
                }
        
        if best_match:
            matches.append(best_match)
    
    return matches


class CrunchbaseMatcher:
    """Main matching class"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.harmonizer = NameHarmonizer()
    
    def run(self, cb_orgs_path: str, cb_people_path: str, output_dir: str = "./results",
            company_threshold: float = 0.85, people_threshold: float = 0.75,
            batch_size: int = 10000, n_processes: int = None):
        """Run the complete matching process"""
        
        if n_processes is None:
            n_processes = mp.cpu_count() - 1
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Connect to database
        connector = PATSTATConnector(self.db_config)
        connector.connect()
        
        try:
            # Load and harmonize Crunchbase data
            logger.info("Loading Crunchbase organizations...")
            cb_orgs = pd.read_csv(cb_orgs_path)
            cb_orgs = cb_orgs[~cb_orgs['roles'].str.contains('investor', na=False)]  # Exclude VCs
            cb_orgs['harmonized_name'] = cb_orgs['name'].apply(self.harmonizer.harmonize_company)
            logger.info(f"Loaded {len(cb_orgs)} organizations")
            
            logger.info("Loading Crunchbase people...")
            cb_people = pd.read_csv(cb_people_path)
            cb_people['name'] = cb_people['first_name'].fillna('') + ' ' + cb_people['last_name'].fillna('')
            cb_people['harmonized_name'] = cb_people['name'].apply(self.harmonizer.harmonize_person)
            logger.info(f"Loaded {len(cb_people)} people")
            
            # Load PATSTAT data
            logger.info("Loading PATSTAT applicants...")
            ps_applicants = connector.get_applicants()
            ps_applicants['harmonized_name'] = ps_applicants['person_name'].apply(self.harmonizer.harmonize_company)
            logger.info(f"Loaded {len(ps_applicants)} applicants")
            
            logger.info("Loading PATSTAT inventors...")
            ps_inventors = connector.get_inventors()
            ps_inventors['harmonized_name'] = ps_inventors['person_name'].apply(self.harmonizer.harmonize_person)
            logger.info(f"Loaded {len(ps_inventors)} inventors")
            
            # Match companies in parallel
            logger.info("Matching companies...")
            company_matches = self._parallel_match(
                cb_orgs, ps_applicants, match_companies_batch, 
                company_threshold, batch_size, n_processes
            )
            
            # Match people in parallel
            logger.info("Matching people...")
            people_matches = self._parallel_match(
                cb_people, ps_inventors, match_people_batch,
                people_threshold, batch_size, n_processes
            )
            
            # Save results
            company_df = pd.DataFrame(company_matches)
            people_df = pd.DataFrame(people_matches)
            
            company_df.to_csv(f"{output_dir}/company_matches.csv", index=False)
            people_df.to_csv(f"{output_dir}/people_matches.csv", index=False)
            
            # Generate summary
            summary = {
                'execution_time': datetime.now().isoformat(),
                'company_matches': len(company_matches),
                'people_matches': len(people_matches),
                'company_match_rate': len(company_matches) / len(cb_orgs) * 100,
                'people_match_rate': len(people_matches) / len(cb_people) * 100
            }
            
            with open(f"{output_dir}/summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Matching complete: {len(company_matches)} companies, {len(people_matches)} people")
            
        finally:
            connector.disconnect()
    
    def _parallel_match(self, cb_data: pd.DataFrame, ps_data: pd.DataFrame,
                       match_func, threshold: float, batch_size: int, n_processes: int) -> List[Dict]:
        """Run matching in parallel"""
        # Split CB data into chunks
        chunks = [cb_data.iloc[i:i+batch_size] for i in range(0, len(cb_data), batch_size)]
        
        # Process chunks in parallel
        all_matches = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(match_func, chunk, ps_data, threshold)
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing batches"):
                matches = future.result()
                all_matches.extend(matches)
        
        return all_matches


# Simple usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python simplified_matcher.py organizations.csv people.csv")
        sys.exit(1)
    
    # Configure your database
    db_config = DatabaseConfig(
        host="",
        port=,
        database="",
        user="",
        password=""
    )

    # Run matching
    matcher = CrunchbaseMatcher(db_config)
    matcher.run(
        cb_orgs_path=sys.argv[1],
        cb_people_path=sys.argv[2],
        output_dir="./matching_results",
        batch_size=10000,  # Adjust based on memory
        n_processes=16  # Adjust based on CPU cores
    )