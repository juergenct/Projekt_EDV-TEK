#!/usr/bin/env python3
import os
import sys
import logging
import warnings
import psycopg2
import pandas as pd
import pickle

# Suppress pandas SQLAlchemy warning for psycopg2 connections
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')
# Suppress pandas deprecation warnings
warnings.filterwarnings('ignore', message='DataFrameGroupBy.apply operated on the grouping columns')
warnings.filterwarnings('ignore', message="'Series.swapaxes' is deprecated")
warnings.filterwarnings('ignore', message="'DataFrame.swapaxes' is deprecated")
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import unicodedata
import re
from rapidfuzz import fuzz, process
from typing import Dict, List, Tuple, Optional, Set
import time
from datetime import datetime
import json
from collections import defaultdict, Counter
from dataclasses import dataclass
import jellyfish
import Levenshtein
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimized_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Global worker functions for multiprocessing (must be picklable)
def harmonize_chunk_company(chunk):
    """Worker function for company name harmonization"""
    return chunk.apply(OptimizedNameHarmonizer.harmonize_company)

def harmonize_chunk_person(chunk):
    """Worker function for person name harmonization"""
    return chunk.apply(OptimizedNameHarmonizer.harmonize_person)


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str

class OptimizedNameHarmonizer:
    """Enhanced name harmonization following paper methodology"""
    
    # Legal designations to remove (comprehensive list from paper)
    LEGAL_TERMS = [
        # English
        r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|llp|lp|plc|pllc)\b',
        r'\b(company|co|companies|comp|cpy)\b',
        r'\b(holdings|holding|group|groups)\b',
        r'\b(partners|partnership|associates|association|assoc|assn)\b',
        r'\b(ventures|venture|capital|investment|investments|invest)\b',
        r'\b(technologies|technology|tech|technical)\b',
        r'\b(solutions|solution|systems|system|services|service|svcs)\b',
        r'\b(international|intl|global|worldwide)\b',
        r'\b(industries|industry|industrial|ind)\b',
        r'\b(enterprises|enterprise|ent)\b',
        r'\b(laboratories|laboratory|labs|lab)\b',
        r'\b(research|development|r&d|rnd)\b',
        r'\b(manufacturing|mfg|products|product|prod)\b',
        r'\b(consulting|consultants|consultant)\b',
        r'\b(management|mgmt|mgt)\b',
        # German
        r'\b(gmbh|ag|kg|ohg|gbr|ug|eg|ev|gmbh & co kg)\b',
        r'\b(gesellschaft|gesellschaft mit beschränkter haftung)\b',
        r'\b(aktiengesellschaft|kommanditgesellschaft)\b',
        # French
        r'\b(sa|sarl|sas|sasu|sci|snc|scs|sca|eurl|ei|eirl)\b',
        r'\b(société|societe|ste|sté)\b',
        # Spanish/Portuguese
        r'\b(sl|sa|sau|slu|sc|scom|scoop)\b',
        r'\b(sociedad|sociedade|ltda|lda)\b',
        # Italian
        r'\b(spa|srl|sas|snc|sapa|ss)\b',
        r'\b(società|societa)\b',
        # Dutch
        r'\b(bv|nv|vof|cv|maatschap)\b',
        r'\b(besloten vennootschap|naamloze vennootschap)\b',
        # Other
        r'\b(oyj|oy)\b',  # Nordic
        r'\b(pte|pvt|pty|proprietary)\b',  # Australia/Singapore
        r'\b(kabushiki kaisha|kk|gk|yugen kaisha|yk)\b',  # Japanese
        r'\b(co ltd|company limited)\b',  # Asian
        r'\b(ooo|oao|zao|pao|nko)\b',  # Russian
        r'\b(sp z oo|spółka z ograniczoną odpowiedzialnością)\b',  # Polish
        r'\b(sro|vos|spol s ro)\b',  # Czech/Slovak
        r'\b(kft|bt|rt|nyrt)\b',  # Hungarian
        r'\b(doo|ad|dd)\b',  # Balkan
        r'\b(tov|dat|pp|fop)\b',  # Ukrainian
        r'\b(sia)\b',  # Baltic
        r'\b(adr|gdr|reit|etf|spac|llc|l3c)\b',  # Financial structures
        r'\b(a/s)\b',  # Nordic
        r'\b(as)\b(?=\s|$)',  # Only match "AS" at end
        r'\b(ap)\b(?=\s|$)',  # Only match "AP" at end
        r'\b(aps)\b(?=\s|$)',  # Only match "APS" at end
        r'\b(is)\b(?=\s|$)',  # Only match "IS" at end
        r'\b(ks)\b(?=\s|$)',  # Only match "KS" at end
        r'\b(hf)\b(?=\s|$)',  # Only match "HF" at end
        r'\b(ehf)\b(?=\s|$)',  # Only match "EHF" at end
    ]
    
    # Academic titles for person names (international)
    TITLES = [
        # Academic
        r'\b(dr|doctor|prof|professor|phd|ph d|dphil|d phil)\b',
        r'\b(md|m d|msc|m sc|ms|m s|ma|m a|mba|m b a)\b',
        r'\b(bsc|b sc|bs|b s|ba|b a|bed|b ed|beng|b eng)\b',
        r'\b(llm|ll m|lld|ll d|jd|j d|esq|esquire)\b',
        r'\b(mphil|m phil|mres|m res|meng|m eng|march|m arch)\b',
        r'\b(dipl|diplom|diploma|dip|cert|certificate)\b',
        # Professional
        r'\b(ing|ingenieur|engineer|eng|ir|arch|architect)\b',
        r'\b(cpa|cfa|cia|cma|acca|ca|chartered accountant)\b',
        r'\b(pe|professional engineer|pmp|prince2)\b',
        # Medical
        r'\b(dds|d d s|dmd|d m d|do|d o|pharmd|pharm d)\b',
        r'\b(rn|r n|np|n p|pa|p a|dvm|d v m)\b',
        # Honorary/Social
        r'\b(mr|mister|mrs|missus|ms|miss|mx|master)\b',
        r'\b(sir|dame|lord|lady|baron|baroness|count|countess)\b',
        r'\b(hon|honorable|rt hon|right honorable)\b',
        # Military
        r'\b(gen|general|lt gen|maj gen|brig gen|col|colonel)\b',
        r'\b(lt col|maj|major|capt|captain|lt|lieutenant)\b',
        r'\b(sgt|sergeant|cpl|corporal|pvt|private)\b',
        # Religious
        r'\b(rev|reverend|fr|father|sr|sister|br|brother)\b',
        r'\b(rabbi|imam|sheikh|pastor|minister)\b',
        # International variations
        r'\b(herr|frau|fraulein|signor|signora|signorina)\b',  # German/Italian
        r'\b(monsieur|m|madame|mme|mademoiselle|mlle)\b',  # French
        r'\b(señor|sr|señora|sra|señorita|srta|don|doña)\b',  # Spanish
        r'\b(senhor|sr|senhora|sra|senhorita|srta|dom|dona)\b',  # Portuguese
    ]
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Convert accented characters to ASCII equivalents"""
        if not text:
            return ""
        
        # Specific character replacements
        replacements = {
            'æ': 'ae', 'Æ': 'AE',
            'œ': 'oe', 'Œ': 'OE',
            'ø': 'o', 'Ø': 'O',
            'ß': 'ss', 'ẞ': 'SS',
            'þ': 'th', 'Þ': 'TH',
            'ð': 'd', 'Ð': 'D',
            'đ': 'd', 'Đ': 'D',
            'ł': 'l', 'Ł': 'L',
            'ı': 'i', 'İ': 'I',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # General normalization
        nfd = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
        
        # Remove any remaining non-ASCII characters
        text = ''.join(char if ord(char) < 128 else ' ' for char in text)
        
        return text
    
    @staticmethod
    def harmonize_company(name: str) -> str:
        """Harmonize company names according to paper specs"""
        if pd.isna(name) or not name:
            return ""
        
        # Convert to uppercase as specified in paper
        name = str(name).upper().strip()
        
        # Normalize unicode characters
        name = OptimizedNameHarmonizer.normalize_unicode(name)
        
        # Remove text in parentheses or brackets
        name = re.sub(r'\([^)]*\)', ' ', name)
        name = re.sub(r'\[[^\]]*\]', ' ', name)
        
        # Remove legal designations - apply in order from longest to shortest
        sorted_terms = sorted(OptimizedNameHarmonizer.LEGAL_TERMS, 
                            key=lambda x: -len(x.replace(r'\b', '').replace('(', '').replace(')', '')))
        
        for pattern in sorted_terms:
            name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)
        
        # Remove common words at the end
        name = re.sub(r'\b(THE|AND|OF|FOR|IN|ON|AT|TO|BY|WITH|FROM)\b', ' ', name)
        
        # Remove punctuation and normalize spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        return name
    
    @staticmethod
    def harmonize_person(name: str) -> str:
        """Harmonize person names according to paper specs"""
        if pd.isna(name) or not name:
            return ""
        
        # Convert to uppercase as specified in paper
        name = str(name).upper().strip()
        
        # Normalize unicode characters
        name = OptimizedNameHarmonizer.normalize_unicode(name)
        
        # Remove c/o company names
        name = re.sub(r'\bC/O\b.*', '', name, flags=re.IGNORECASE)
        
        # Remove titles
        for pattern in OptimizedNameHarmonizer.TITLES:
            name = re.sub(pattern.upper(), ' ', name, flags=re.IGNORECASE)
        
        # Remove text in parentheses
        name = re.sub(r'\([^)]*\)', ' ', name)
        
        # Remove middle initials (single letters followed by period or space)
        name = re.sub(r'\b[A-Z]\b\.?', ' ', name)
        
        # Remove punctuation and normalize spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        return name
    
    @staticmethod
    def get_alphanumeric_only(name: str) -> str:
        """Keep only [A-Z] and [0-9] for alphanumeric matching"""
        if pd.isna(name) or not name:
            return ""
        return re.sub(r'[^A-Z0-9]', '', str(name).upper())


class OptimizedStagedMatcher:
    """Implements 4-stage matching as described in the paper"""
    
    @staticmethod
    def perfect_match(str1: str, str2: str) -> bool:
        """Stage 1: Perfect match after harmonization"""
        return str1 == str2 and str1 != ""
    
    @staticmethod
    def alphanumeric_match(str1: str, str2: str) -> bool:
        """Stage 2: Alphanumeric match (e.g., I.B.M. = IBM = I B M)"""
        alphanum1 = OptimizedNameHarmonizer.get_alphanumeric_only(str1)
        alphanum2 = OptimizedNameHarmonizer.get_alphanumeric_only(str2)
        return alphanum1 == alphanum2 and alphanum1 != ""
    
    @staticmethod
    def jaro_winkler_score(str1: str, str2: str, threshold: float = 0.85) -> float:
        """Stage 3: Jaro-Winkler similarity with token weighting"""
        if not str1 or not str2:
            return 0.0
        
        # Basic Jaro-Winkler score
        basic_score = jellyfish.jaro_winkler_similarity(str1, str2)
        
        if basic_score < threshold:
            return 0.0
        
        # Token-based weighting as mentioned in paper
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        
        if not tokens1 or not tokens2:
            return basic_score
        
        # Calculate token overlap
        common_tokens = tokens1 & tokens2
        all_tokens = tokens1 | tokens2
        
        if not all_tokens:
            return basic_score
        
        # Simple token overlap
        token_score = len(common_tokens) / len(all_tokens)
        
        # Combine scores
        return 0.7 * basic_score + 0.3 * token_score
    
    @staticmethod
    def levenshtein_score(str1: str, str2: str, max_distance: int = 3) -> float:
        """Stage 4: Levenshtein distance (normalized)"""
        if not str1 or not str2:
            return 0.0
        
        distance = Levenshtein.distance(str1, str2)
        
        # As per paper, Levenshtein can cause false positives with acronyms
        if distance > max_distance:
            return 0.0
        
        max_len = max(len(str1), len(str2))
        return 1 - (distance / max_len) if max_len > 0 else 0.0
    
    @staticmethod
    def ngram_similarity(str1: str, str2: str, n: int = 2) -> float:
        """2-gram similarity as described in paper for people matching"""
        if len(str1) < n or len(str2) < n:
            return 0.0
        
        # Get n-grams
        ngrams1 = [str1[i:i+n] for i in range(len(str1) - n + 1)]
        ngrams2 = [str2[i:i+n] for i in range(len(str2) - n + 1)]
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Count common n-grams
        common_count = 0
        ngrams2_copy = ngrams2.copy()
        
        for ng in ngrams1:
            if ng in ngrams2_copy:
                common_count += 1
                ngrams2_copy.remove(ng)  # Remove to handle duplicates correctly
        
        # Calculate similarity as in paper
        total_ngrams = len(ngrams1) + len(ngrams2)
        
        return (2 * common_count) / total_ngrams if total_ngrams > 0 else 0.0


class OptimizedCrunchbasePATSTATMatcher:
    """
    Optimized implementation that performs heavy computation in Python
    using multiprocessing instead of PostgreSQL
    """
    
    def __init__(self, patstat_config: dict = None, num_processes: int = None, chunk_size: int = 10000, 
                 checkpoint_dir: str = '/mnt/hdd02/Projekt_EDV_TEK/matching_checkpoints',
                 hpc_mode: bool = False, hpc_data_dir: str = '/fibus/fs1/0f/cyh1826/wt/edv_tek/'):
        self.patstat_config = patstat_config
        self.num_processes = num_processes or (32 if hpc_mode else 16)
        self.chunk_size = chunk_size
        self.harmonizer = OptimizedNameHarmonizer()
        self.hpc_mode = hpc_mode
        self.hpc_data_dir = hpc_data_dir
        
        # Set checkpoint directory based on mode
        if hpc_mode:
            self.checkpoint_dir = os.path.join(hpc_data_dir, 'matching_checkpoints')
        else:
            self.checkpoint_dir = checkpoint_dir
            
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        mode_str = "HPC" if hpc_mode else "LOCAL"
        logger.info(f"Initialized in {mode_str} mode with {self.num_processes} processes, chunk size {chunk_size}")
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        
        if hpc_mode:
            logger.info(f"HPC data directory: {self.hpc_data_dir}")
            if patstat_config is None:
                logger.info("Running in HPC mode - PostgreSQL operations will be skipped")
    
    def save_checkpoint(self, milestone_name: str, data: dict, description: str = ""):
        """Save data checkpoint to disk"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{milestone_name}.pkl")
        
        logger.info(f"💾 SAVING CHECKPOINT: {milestone_name}")
        if description:
            logger.info(f"   Description: {description}")
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save as CSV for human inspection
        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                csv_path = os.path.join(self.checkpoint_dir, f"{milestone_name}_{key}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"   Saved {key}: {len(df):,} records → {csv_path}")
        
        logger.info(f"   Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, milestone_name: str) -> Optional[dict]:
        """Load data checkpoint from disk"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{milestone_name}.pkl")
        
        if os.path.exists(checkpoint_path):
            logger.info(f"📂 LOADING CHECKPOINT: {milestone_name}")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            for key, df in data.items():
                if isinstance(df, pd.DataFrame):
                    logger.info(f"   Loaded {key}: {len(df):,} records")
            
            logger.info(f"   Checkpoint loaded: {checkpoint_path}")
            return data
        else:
            logger.info(f"   No checkpoint found: {checkpoint_path}")
            return None
    
    def extract_patstat_data(self, cleantech_appln_ids: Optional[pd.DataFrame] = None,
                           cleantech_person_ids: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """Extract all necessary data from PATSTAT in bulk - minimal SQL, maximum Python processing"""
        
        # 🏁 MILESTONE 1: Check if raw PATSTAT data already extracted
        milestone_name = "01_raw_patstat_data"
        checkpoint = self.load_checkpoint(milestone_name)
        if checkpoint is not None:
            logger.info("✅ Using cached PATSTAT data from checkpoint")
            return checkpoint
        
        logger.info("🔄 MILESTONE 1: Extracting PATSTAT data from PostgreSQL...")
        
        # Log filtering status
        if cleantech_appln_ids is not None and cleantech_person_ids is not None:
            logger.info(f"✓ Using CLEANTECH FILTERS:")
            logger.info(f"  - {len(cleantech_appln_ids):,} cleantech application IDs")
            logger.info(f"  - {len(cleantech_person_ids):,} cleantech person IDs")
            logger.info("  → This will significantly reduce data volume and runtime!")
        elif cleantech_appln_ids is not None:
            logger.info(f"✓ Using partial cleantech filter: {len(cleantech_appln_ids):,} application IDs")
        elif cleantech_person_ids is not None:
            logger.info(f"✓ Using partial cleantech filter: {len(cleantech_person_ids):,} person IDs")
        else:
            logger.warning("⚠ NO CLEANTECH FILTERS - Processing ALL patents (this will be slow!)")
        
        with psycopg2.connect(**self.patstat_config) as conn:
            # Create INDEXED temporary tables for efficient filtering
            if cleantech_appln_ids is not None:
                logger.info("Creating indexed cleantech application filter table...")
                with conn.cursor() as cur:
                    cur.execute("CREATE TEMP TABLE temp_cleantech_appln_ids (appln_id INTEGER PRIMARY KEY)")
                    
                    # Use COPY for ultra-fast bulk loading
                    from io import StringIO
                    buffer = StringIO()
                    cleantech_appln_ids[['appln_id']].to_csv(buffer, index=False, header=False)
                    buffer.seek(0)
                    cur.copy_from(buffer, 'temp_cleantech_appln_ids', columns=['appln_id'])
                    conn.commit()
                    logger.info(f"Indexed {len(cleantech_appln_ids):,} cleantech application IDs")
            
            if cleantech_person_ids is not None:
                logger.info("Creating indexed cleantech person filter table...")
                with conn.cursor() as cur:
                    cur.execute("CREATE TEMP TABLE temp_cleantech_person_ids (person_id INTEGER PRIMARY KEY)")
                    
                    # Use COPY for ultra-fast bulk loading
                    from io import StringIO
                    buffer = StringIO()
                    cleantech_person_ids[['person_id']].to_csv(buffer, index=False, header=False)
                    buffer.seek(0)
                    cur.copy_from(buffer, 'temp_cleantech_person_ids', columns=['person_id'])
                    conn.commit()
                    logger.info(f"Indexed {len(cleantech_person_ids):,} cleantech person IDs")
            
            # Extract raw data with INNER JOINs for maximum efficiency
            # 1. Person-application data - filter as early as possible
            if cleantech_appln_ids is not None and cleantech_person_ids is not None:
                # Most efficient: filter by both person AND application
                query = """
                    SELECT 
                        p.person_id,
                        pa.appln_id,
                        p.person_name,
                        p.person_ctry_code,
                        p.person_address,
                        pa.applt_seq_nr,
                        pa.invt_seq_nr
                    FROM temp_cleantech_person_ids tcp
                    INNER JOIN tls206_person p ON CAST(p.person_id AS INTEGER) = tcp.person_id
                    INNER JOIN tls207_pers_appln pa ON p.person_id = pa.person_id
                    INNER JOIN temp_cleantech_appln_ids tca ON CAST(pa.appln_id AS INTEGER) = tca.appln_id
                """
            elif cleantech_person_ids is not None:
                # Filter by person only
                query = """
                    SELECT 
                        p.person_id,
                        pa.appln_id,
                        p.person_name,
                        p.person_ctry_code,
                        p.person_address,
                        pa.applt_seq_nr,
                        pa.invt_seq_nr
                    FROM temp_cleantech_person_ids tcp
                    INNER JOIN tls206_person p ON CAST(p.person_id AS INTEGER) = tcp.person_id
                    INNER JOIN tls207_pers_appln pa ON p.person_id = pa.person_id
                """
            elif cleantech_appln_ids is not None:
                # Filter by application only
                query = """
                    SELECT 
                        p.person_id,
                        pa.appln_id,
                        p.person_name,
                        p.person_ctry_code,
                        p.person_address,
                        pa.applt_seq_nr,
                        pa.invt_seq_nr
                    FROM temp_cleantech_appln_ids tca
                    INNER JOIN tls207_pers_appln pa ON CAST(pa.appln_id AS INTEGER) = tca.appln_id
                    INNER JOIN tls206_person p ON p.person_id = pa.person_id
                """
            else:
                # No filters - original query
                query = """
                    SELECT 
                        p.person_id,
                        pa.appln_id,
                        p.person_name,
                        p.person_ctry_code,
                        p.person_address,
                        pa.applt_seq_nr,
                        pa.invt_seq_nr
                    FROM tls206_person p
                    JOIN tls207_pers_appln pa ON p.person_id = pa.person_id
                """
            
            logger.info("📊 Fetching person-application data...")
            pers_appln_df = pd.read_sql(query, conn)
            logger.info(f"   → Loaded {len(pers_appln_df):,} person-application records")
            
            # 2. Application data - only for cleantech applications
            if cleantech_appln_ids is not None:
                # Efficient: start with cleantech filter
                query = """
                    SELECT 
                        a.appln_id,
                        a.appln_filing_year,
                        a.appln_auth,
                        a.docdb_family_id
                    FROM temp_cleantech_appln_ids tca
                    INNER JOIN tls201_appln a ON CAST(a.appln_id AS INTEGER) = tca.appln_id
                    WHERE a.appln_auth IN ('EP', 'US')
                """
            else:
                # No filter - original query
                query = """
                    SELECT 
                        appln_id,
                        appln_filing_year,
                        appln_auth,
                        docdb_family_id
                    FROM tls201_appln
                    WHERE appln_auth IN ('EP', 'US')
                """
            
            logger.info("📊 Fetching application data...")
            appln_df = pd.read_sql(query, conn)
            logger.info(f"   → Loaded {len(appln_df):,} application records")
            
            # 3. IPC codes - only for cleantech applications
            if cleantech_appln_ids is not None:
                # Efficient: start with cleantech filter
                query = """
                    SELECT 
                        i.appln_id,
                        i.ipc_class_symbol
                    FROM temp_cleantech_appln_ids tca
                    INNER JOIN tls209_appln_ipc i ON CAST(i.appln_id AS INTEGER) = tca.appln_id
                """
            else:
                # No filter - original query
                query = """
                    SELECT 
                        appln_id,
                        ipc_class_symbol
                    FROM tls209_appln_ipc
                """
            
            logger.info("📊 Fetching IPC codes...")
            ipc_df = pd.read_sql(query, conn)
            logger.info(f"   → Loaded {len(ipc_df):,} IPC code records")
        
        # Now do ALL processing in Python
        logger.info("🔄 Processing data in Python...")
        
        # Convert text columns to integers
        logger.info("   → Converting data types...")
        for df in [pers_appln_df, appln_df, ipc_df]:
            for col in df.columns:
                if col in ['person_id', 'appln_id', 'applt_seq_nr', 'invt_seq_nr', 
                          'appln_filing_year', 'docdb_family_id']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Process IPC codes - extract first 4 characters in Python
        logger.info("   → Processing IPC codes...")
        ipc_df['ipc4_code'] = ipc_df['ipc_class_symbol'].str[:4]
        ipc_df = ipc_df[['appln_id', 'ipc4_code']].drop_duplicates()
        
        # Merge person-application with application data
        logger.info("   → Merging person-application with application data...")
        merged_df = pers_appln_df.merge(appln_df, on='appln_id', how='inner')
        
        # Split into applicants and inventors
        logger.info("   → Splitting into applicants and inventors...")
        applicants_df = merged_df[merged_df['applt_seq_nr'] > 0].copy()
        inventors_df = merged_df[merged_df['invt_seq_nr'] > 0].copy()
        
        # Remove duplicates
        logger.info("   → Removing duplicates...")
        applicants_df = applicants_df.drop_duplicates(subset=['person_id', 'appln_id'])
        inventors_df = inventors_df.drop_duplicates(subset=['person_id', 'appln_id'])
        
        logger.info(f"Extracted {len(applicants_df):,} applicant records")
        logger.info(f"Extracted {len(inventors_df):,} inventor records")
        logger.info(f"Extracted {len(ipc_df):,} unique IPC codes")
        
        # Log efficiency gains if using cleantech filters
        if cleantech_appln_ids is not None or cleantech_person_ids is not None:
            logger.info("✓ Cleantech filtering significantly reduced data volume!")
            total_records = len(applicants_df) + len(inventors_df)
            logger.info(f"  → Processing only {total_records:,} person-application records (vs millions without filtering)")
        
        # 💾 MILESTONE 1: Save raw PATSTAT data
        patstat_data = {
            'applicants': applicants_df,
            'inventors': inventors_df,
            'ipc': ipc_df
        }
        self.save_checkpoint(
            "01_raw_patstat_data", 
            patstat_data,
            "Raw PATSTAT data extracted from PostgreSQL"
        )
        
        return patstat_data
    
    def extract_crunchbase_data(self, org_csv_path: str = None, people_csv_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract Crunchbase data from CSV files"""
        
        # 🏁 MILESTONE 2: Check if Crunchbase data already loaded
        milestone_name = "02_raw_crunchbase_data"
        checkpoint = self.load_checkpoint(milestone_name)
        if checkpoint is not None:
            logger.info("✅ Using cached Crunchbase data from checkpoint")
            return checkpoint['companies'], checkpoint['people']
        
        # In HPC mode, use the same file names from local machine
        if self.hpc_mode and (org_csv_path is None or people_csv_path is None):
            logger.info("🔍 HPC mode: Using same file names as local machine...")
            
            if org_csv_path is None:
                org_csv_path = os.path.join(self.hpc_data_dir, 'organizations.csv')
                logger.info(f"   Organizations file: {org_csv_path}")
                        
            if people_csv_path is None:
                people_csv_path = os.path.join(self.hpc_data_dir, 'people.csv')
                logger.info(f"   People file: {people_csv_path}")
            
            # Verify files exist
            if not os.path.exists(org_csv_path) or not os.path.exists(people_csv_path):
                raise FileNotFoundError(f"Crunchbase CSV files not found in {self.hpc_data_dir}. Expected: organizations.csv and people.csv")
        
        logger.info("🔄 MILESTONE 2: Loading Crunchbase data from CSV files...")
        
        # Load organizations
        logger.info(f"Loading organizations from: {org_csv_path}")
        companies_df = pd.read_csv(org_csv_path)
        logger.info(f"Loaded {len(companies_df)} organizations")
        
        # Load people  
        logger.info(f"Loading people from: {people_csv_path}")
        people_df = pd.read_csv(people_csv_path)
        logger.info(f"Loaded {len(people_df)} people")
        
        # Process in Python
        logger.info("Processing Crunchbase data...")
        
        # Remove duplicates if UUID column exists
        if 'uuid' in companies_df.columns:
            companies_df = companies_df.drop_duplicates(subset=['uuid'])
        
        if 'uuid' in people_df.columns:
            people_df = people_df.drop_duplicates(subset=['uuid'])
        
        # Create full name if needed
        if 'name' not in people_df.columns or people_df['name'].isna().all():
            if 'first_name' in people_df.columns and 'last_name' in people_df.columns:
                people_df['name'] = (people_df['first_name'].fillna('') + ' ' + 
                                   people_df['last_name'].fillna('')).str.strip()
        
        # Ensure required columns exist
        required_org_cols = ['name']
        required_people_cols = ['name']
        
        for col in required_org_cols:
            if col not in companies_df.columns:
                raise ValueError(f"Required column '{col}' not found in organizations CSV")
                
        for col in required_people_cols:
            if col not in people_df.columns:
                raise ValueError(f"Required column '{col}' not found in people CSV")
        
        logger.info(f"Processed {len(companies_df)} unique organizations")
        logger.info(f"Processed {len(people_df)} unique people")
        
        # 💾 MILESTONE 2: Save raw Crunchbase data
        crunchbase_data = {
            'companies': companies_df,
            'people': people_df
        }
        self.save_checkpoint(
            "02_raw_crunchbase_data",
            crunchbase_data,
            "Raw Crunchbase data loaded from CSV files"
        )
        
        return companies_df, people_df
    
    def refill_country_codes_batch(self, df: pd.DataFrame, df_type: str = 'patstat') -> pd.DataFrame:
        """Refill missing country codes using paper methodology - optimized for batch processing"""
        missing_before = df['person_ctry_code' if df_type == 'patstat' else 'country_code'].isna().sum()
        logger.info(f"Refilling {df_type} country codes... ({missing_before:,} missing)")
        
        if df_type == 'patstat':
            # Group by person_name to find homonyms
            name_groups = df.groupby('person_name')['person_ctry_code'].apply(
                lambda x: x.mode()[0] if len(x.mode()) > 0 and pd.notna(x.mode()[0]) else np.nan
            )
            
            # Fill missing countries with modal country for same name
            missing_mask = df['person_ctry_code'].isna()
            df.loc[missing_mask, 'person_ctry_code'] = df.loc[missing_mask, 'person_name'].map(name_groups)
            
            # For remaining missing, use patent office nationality
            still_missing = df['person_ctry_code'].isna()
            df.loc[still_missing & (df['appln_auth'] == 'US'), 'person_ctry_code'] = 'US'
            df.loc[still_missing & (df['appln_auth'] == 'EP'), 'person_ctry_code'] = 'EP'
            
        else:  # crunchbase
            # Use phone country codes
            if 'phone_number' in df.columns:
                phone_country_map = {
                    '+1': 'US', '+44': 'GB', '+49': 'DE', '+33': 'FR', '+39': 'IT',
                    '+34': 'ES', '+31': 'NL', '+41': 'CH', '+46': 'SE', '+47': 'NO',
                    '+45': 'DK', '+358': 'FI', '+43': 'AT', '+32': 'BE', '+351': 'PT',
                    '+353': 'IE', '+48': 'PL', '+420': 'CZ', '+36': 'HU', '+30': 'GR',
                    '+90': 'TR', '+7': 'RU', '+380': 'UA', '+40': 'RO', '+359': 'BG',
                    '+386': 'SI', '+385': 'HR', '+381': 'RS', '+370': 'LT', '+371': 'LV',
                    '+372': 'EE', '+374': 'AM', '+375': 'BY', '+994': 'AZ', '+995': 'GE',
                    '+86': 'CN', '+81': 'JP', '+82': 'KR', '+91': 'IN', '+62': 'ID',
                    '+60': 'MY', '+65': 'SG', '+66': 'TH', '+84': 'VN', '+63': 'PH',
                    '+61': 'AU', '+64': 'NZ', '+27': 'ZA', '+20': 'EG', '+212': 'MA',
                    '+216': 'TN', '+213': 'DZ', '+234': 'NG', '+254': 'KE', '+255': 'TZ',
                    '+256': 'UG', '+233': 'GH', '+237': 'CM', '+225': 'CI', '+221': 'SN',
                    '+52': 'MX', '+54': 'AR', '+55': 'BR', '+56': 'CL', '+57': 'CO',
                    '+58': 'VE', '+51': 'PE', '+593': 'EC', '+598': 'UY', '+595': 'PY',
                    '+591': 'BO', '+507': 'PA', '+506': 'CR', '+503': 'SV', '+502': 'GT',
                    '+504': 'HN', '+505': 'NI', '+972': 'IL', '+966': 'SA', '+971': 'AE',
                    '+974': 'QA', '+965': 'KW', '+968': 'OM', '+973': 'BH', '+961': 'LB',
                    '+962': 'JO', '+963': 'SY', '+964': 'IQ', '+98': 'IR', '+92': 'PK',
                    '+93': 'AF', '+94': 'LK', '+95': 'MM', '+977': 'NP', '+880': 'BD',
                }
                
                def extract_country_from_phone(phone):
                    if pd.isna(phone):
                        return np.nan
                    phone_str = str(phone)
                    for prefix, country in sorted(phone_country_map.items(), key=lambda x: -len(x[0])):
                        if phone_str.startswith(prefix):
                            return country
                    return np.nan
                
                missing_mask = df['country_code'].isna()
                df.loc[missing_mask, 'country_code'] = df.loc[missing_mask, 'phone_number'].apply(extract_country_from_phone)
        
        remaining_missing = df['person_ctry_code' if df_type == 'patstat' else 'country_code'].isna().sum()
        logger.info(f"After refilling: {remaining_missing} missing country codes")
        
        return df
    
    def harmonize_names_parallel(self, names: pd.Series, name_type: str = 'company') -> pd.Series:
        """Harmonize names in parallel with progress bar"""
        logger.info(f"Harmonizing {len(names)} {name_type} names using {self.num_processes} processes...")
        
        # Choose worker function
        if name_type == 'company':
            worker_func = harmonize_chunk_company
        else:
            worker_func = harmonize_chunk_person
        
        # Handle edge cases
        if len(names) == 0:
            return pd.Series([], dtype=str)
        
        if len(names) <= self.num_processes:
            # Too few items for parallel processing
            logger.info("Processing sequentially due to small dataset...")
            if name_type == 'company':
                tqdm.pandas(desc=f"Harmonizing {name_type} names")
                return names.progress_apply(OptimizedNameHarmonizer.harmonize_company)
            else:
                tqdm.pandas(desc=f"Harmonizing {name_type} names")
                return names.progress_apply(OptimizedNameHarmonizer.harmonize_person)
        
        # Split into chunks for parallel processing
        chunk_size = max(1, len(names) // self.num_processes + 1)
        chunks = [names[i:i + chunk_size] for i in range(0, len(names), chunk_size)]
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 0]
        
        logger.info(f"Processing {len(chunks)} chunks in parallel...")
        
        with Pool(self.num_processes) as pool:
            # Use tqdm to track progress of parallel processing
            with tqdm(total=len(chunks), desc=f"Harmonizing {name_type} names (chunks)") as pbar:
                results = []
                for result in pool.imap(worker_func, chunks):
                    results.append(result)
                    pbar.update(1)
        
        # Handle empty results
        if not results:
            return pd.Series([], dtype=str)
        
        logger.info("Concatenating results...")
        return pd.concat(results, ignore_index=True)
    
    def disambiguate_inventors_optimized(self, inventors_df: pd.DataFrame, ipc_df: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate PATSTAT inventors using paper's 5 criteria - optimized version"""
        logger.info("Disambiguating inventors...")
        
        # Add harmonized names
        inventors_df['harmonized_name'] = self.harmonize_names_parallel(inventors_df['person_name'], 'person')
        
        # Merge IPC codes
        inventors_with_ipc = inventors_df.merge(ipc_df, on='appln_id', how='left')
        
        # Create inventor features DataFrame
        inventor_features = inventors_with_ipc.groupby('person_id').agg({
            'person_name': 'first',
            'harmonized_name': 'first',
            'person_ctry_code': lambda x: x.mode()[0] if len(x.mode()) > 0 else '',
            'docdb_family_id': lambda x: list(set(x)),
            'appln_id': lambda x: list(set(x)),
            'ipc4_code': lambda x: list(set(x.dropna())),
            'appln_filing_year': lambda x: list(x)
        }).reset_index()
        
        # Group by harmonized name for disambiguation
        name_groups = inventor_features.groupby('harmonized_name')
        
        unified_mapping = {}
        unified_id_counter = 1
        
        logger.info(f"Disambiguating {len(name_groups)} unique harmonized names...")
        
        # Add progress bar for disambiguation
        for name, group in tqdm(name_groups, desc="Disambiguating inventors"):
            if len(group) == 1:
                # Single person with this name
                unified_mapping[group.iloc[0]['person_id']] = unified_id_counter
                unified_id_counter += 1
                continue
            
            # Multiple people with same name - check disambiguation criteria
            group_list = group.to_dict('records')
            clusters = []
            processed = set()
            
            for i, person1 in enumerate(group_list):
                if person1['person_id'] in processed:
                    continue
                
                cluster = [person1['person_id']]
                processed.add(person1['person_id'])
                
                for j, person2 in enumerate(group_list[i+1:], i+1):
                    if person2['person_id'] in processed:
                        continue
                    
                    # Check 5 criteria
                    criteria_met = 0
                    
                    # 1. Common patent family
                    if set(person1['docdb_family_id']) & set(person2['docdb_family_id']):
                        criteria_met += 1
                    
                    # 2. Common IPC4 code
                    if person1['ipc4_code'] and person2['ipc4_code']:
                        if set(person1['ipc4_code']) & set(person2['ipc4_code']):
                            criteria_met += 1
                    
                    # 3. Same country
                    if person1['person_ctry_code'] and person2['person_ctry_code']:
                        if person1['person_ctry_code'] == person2['person_ctry_code']:
                            criteria_met += 1
                    
                    # 4. Overlapping time period
                    if person1['appln_filing_year'] and person2['appln_filing_year']:
                        years1 = set(person1['appln_filing_year'])
                        years2 = set(person2['appln_filing_year'])
                        if years1 & years2:
                            criteria_met += 1
                    
                    # 5. Common co-inventors (simplified - check if they share patents)
                    if len(set(person1['appln_id']) & set(person2['appln_id'])) > 0:
                        criteria_met += 1
                    
                    # Need at least 3 criteria
                    if criteria_met >= 3:
                        cluster.append(person2['person_id'])
                        processed.add(person2['person_id'])
                
                clusters.append(cluster)
            
            # Assign unified IDs
            for cluster in clusters:
                for pid in cluster:
                    unified_mapping[pid] = unified_id_counter
                unified_id_counter += 1
        
        # Add unified ID to dataframe
        inventors_df['unified_person_id'] = inventors_df['person_id'].map(unified_mapping)
        
        logger.info(f"Disambiguation complete: {len(inventors_df)} → {inventors_df['unified_person_id'].nunique()} unique inventors")
        
        return inventors_df
    
    def compute_company_matches_chunk(self, chunk_data: Tuple[pd.DataFrame, pd.DataFrame, str, Dict]) -> List[Dict]:
        """Compute company matches for a chunk of data"""
        cb_chunk, patstat_df, method, company_inventors = chunk_data
        matches = []
        
        for _, cb_row in cb_chunk.iterrows():
            cb_name = cb_row['harmonized_name']
            cb_alphanum = cb_row.get('alphanumeric_name', '')
            cb_country = cb_row.get('country_code', '')
            
            if not cb_name:
                continue
            
            # Filter by country if available
            if pd.notna(cb_country) and cb_country:
                candidates = patstat_df[patstat_df['person_ctry_code'] == cb_country]
            else:
                candidates = patstat_df
            
            if method == 'perfect':
                # Perfect match
                ps_matches = candidates[candidates['harmonized_name'] == cb_name]
            elif method == 'alphanumeric' and cb_alphanum:
                # Alphanumeric match
                ps_matches = candidates[candidates['alphanumeric_name'] == cb_alphanum]
            elif method == 'jaro_winkler':
                # Jaro-Winkler matching
                ps_matches = []
                for _, ps_row in candidates.iterrows():
                    score = OptimizedStagedMatcher.jaro_winkler_score(cb_name, ps_row['harmonized_name'])
                    if score > 0:
                        ps_matches.append((score, ps_row))
                
                # Take best matches
                if ps_matches:
                    ps_matches.sort(key=lambda x: x[0], reverse=True)
                    best_score = ps_matches[0][0]
                    ps_matches = [row for score, row in ps_matches if score >= best_score - 0.05]
                else:
                    ps_matches = pd.DataFrame()
            elif method == 'levenshtein':
                # Levenshtein matching (requires validation)
                ps_matches = []
                for _, ps_row in candidates.iterrows():
                    score = OptimizedStagedMatcher.levenshtein_score(cb_name, ps_row['harmonized_name'])
                    if score > 0.7:
                        # Validate with inventors if available
                        ps_person_id = ps_row['person_id']
                        if ps_person_id in company_inventors:
                            # Simple validation: check if any inventor name is similar
                            validation_score = 0
                            for inv in company_inventors[ps_person_id]:
                                inv_harmonized = self.harmonizer.harmonize_person(inv['name'])
                                if inv_harmonized and cb_name:
                                    sim = fuzz.ratio(cb_name, inv_harmonized) / 100.0
                                    validation_score = max(validation_score, sim)
                            
                            if validation_score > 0.5:
                                final_score = score * 0.6 + validation_score * 0.4
                                ps_matches.append((final_score, ps_row, validation_score))
                
                if ps_matches:
                    ps_matches.sort(key=lambda x: x[0], reverse=True)
                    ps_matches = [(final_score, row, val_score) for final_score, row, val_score in ps_matches[:1]]  # Take best
                else:
                    ps_matches = []
            else:
                continue
            
            # Process matches based on method
            if method in ['perfect', 'alphanumeric'] and isinstance(ps_matches, pd.DataFrame) and not ps_matches.empty:
                for _, ps_row in ps_matches.iterrows():
                    matches.append({
                        'cb_uuid': cb_row['uuid'],
                        'cb_name': cb_row['name'],
                        'cb_harmonized': cb_name,
                        'ps_person_id': ps_row['person_id'],
                        'ps_name': ps_row['person_name'],
                        'ps_harmonized': ps_row['harmonized_name'],
                        'country_code': cb_country,
                        'match_score': 1.0 if method == 'perfect' else 0.9,
                        'match_type': method,
                        'match_stage': 1 if method == 'perfect' else 2
                    })
            elif method == 'jaro_winkler' and isinstance(ps_matches, list):
                for ps_row in ps_matches:
                    if isinstance(ps_row, tuple):
                        score, ps_row = ps_row
                    matches.append({
                        'cb_uuid': cb_row['uuid'],
                        'cb_name': cb_row['name'],
                        'cb_harmonized': cb_name,
                        'ps_person_id': ps_row['person_id'],
                        'ps_name': ps_row['person_name'],
                        'ps_harmonized': ps_row['harmonized_name'],
                        'country_code': cb_country,
                        'match_score': score * 0.8,  # Weighted as per paper
                        'match_type': 'jaro_winkler',
                        'match_stage': 3
                    })
            elif method == 'levenshtein' and isinstance(ps_matches, list):
                for final_score, ps_row, val_score in ps_matches:
                    matches.append({
                        'cb_uuid': cb_row['uuid'],
                        'cb_name': cb_row['name'],
                        'cb_harmonized': cb_name,
                        'ps_person_id': ps_row['person_id'],
                        'ps_name': ps_row['person_name'],
                        'ps_harmonized': ps_row['harmonized_name'],
                        'country_code': cb_country,
                        'match_score': final_score,
                        'match_type': 'levenshtein_validated',
                        'match_stage': 4,
                        'validation_score': val_score
                    })
        
        return matches
    
    def perform_company_matching_staged(self, cb_companies: pd.DataFrame, ps_applicants: pd.DataFrame,
                                      company_inventors: Dict) -> pd.DataFrame:
        """Perform staged company matching using multiprocessing"""
        all_matches = []
        
        # Add harmonized and alphanumeric names
        logger.info("Harmonizing company names...")
        cb_companies['harmonized_name'] = self.harmonize_names_parallel(cb_companies['name'], 'company')
        # Fill any NaN values that might result from harmonization
        cb_companies['harmonized_name'] = cb_companies['harmonized_name'].fillna('')
        cb_companies['alphanumeric_name'] = cb_companies['harmonized_name'].apply(self.harmonizer.get_alphanumeric_only)
        
        ps_applicants['harmonized_name'] = self.harmonize_names_parallel(ps_applicants['person_name'], 'company')
        # Fill any NaN values that might result from harmonization
        ps_applicants['harmonized_name'] = ps_applicants['harmonized_name'].fillna('')
        ps_applicants['alphanumeric_name'] = ps_applicants['harmonized_name'].apply(self.harmonizer.get_alphanumeric_only)
        
        # Track already matched to avoid duplicates
        matched_cb_uuids = set()
        
        # Process each stage
        for method in ['perfect', 'alphanumeric', 'jaro_winkler', 'levenshtein']:
            logger.info(f"Performing {method} matching for companies...")
            
            # Filter out already matched
            remaining_cb = cb_companies[~cb_companies['uuid'].isin(matched_cb_uuids)]
            
            if len(remaining_cb) == 0:
                logger.info(f"No remaining companies for {method} matching")
                continue
            
            # Split into chunks - use pandas-friendly approach
            num_chunks = min(self.num_processes * 4, len(remaining_cb))
            chunk_size = max(1, len(remaining_cb) // num_chunks + 1)
            chunks = [remaining_cb.iloc[i:i + chunk_size] for i in range(0, len(remaining_cb), chunk_size)]
            chunk_data = [(chunk, ps_applicants, method, company_inventors) 
                         for chunk in chunks if len(chunk) > 0]
            
            logger.info(f"Processing {len(remaining_cb):,} companies in {len(chunk_data)} chunks...")
            
            # Process chunks in parallel with progress bar
            with Pool(self.num_processes) as pool:
                with tqdm(total=len(chunk_data), desc=f"Company {method} matching") as pbar:
                    chunk_results = []
                    for result in pool.imap(self.compute_company_matches_chunk, chunk_data):
                        chunk_results.append(result)
                        pbar.update(1)
            
            # Flatten results
            method_matches = [match for chunk in chunk_results for match in chunk]
            logger.info(f"Found {len(method_matches)} {method} matches")
            
            # Update matched set
            for match in method_matches:
                matched_cb_uuids.add(match['cb_uuid'])
            
            all_matches.extend(method_matches)
        
        return pd.DataFrame(all_matches)
    
    def compute_people_matches_chunk(self, chunk_data: Tuple[pd.DataFrame, pd.DataFrame]) -> List[Dict]:
        """Compute people matches for a chunk using 2-gram similarity"""
        cb_chunk, inventors_df = chunk_data
        matches = []
        
        for _, cb_row in cb_chunk.iterrows():
            cb_name = cb_row['harmonized_name']
            
            if not cb_name or len(cb_name) < 3:
                continue
            
            # Initial blocking on first two characters of last name
            cb_tokens = cb_name.split()
            if not cb_tokens:
                continue
            
            last_token = cb_tokens[-1]
            if len(last_token) < 2:
                continue
            
            first_two = last_token[:2]
            
            # Find candidates
            candidates = inventors_df[
                inventors_df['harmonized_name'].str.contains(first_two, na=False, regex=False)
            ]
            
            if candidates.empty:
                continue
            
            # Find best match using 2-gram similarity
            best_score = 0
            best_match = None
            
            for _, inv_row in candidates.iterrows():
                inv_name = inv_row['harmonized_name']
                if not inv_name:
                    continue
                
                # Calculate 2-gram similarity
                score = OptimizedStagedMatcher.ngram_similarity(cb_name, inv_name, n=2)
                
                if score > best_score and score >= 0.75:  # Threshold from paper
                    best_score = score
                    best_match = inv_row
            
            if best_match is not None:
                matches.append({
                    'cb_uuid': cb_row['uuid'],
                    'cb_name': cb_row['name'],
                    'cb_harmonized': cb_name,
                    'ps_person_id': best_match['person_id'],
                    'ps_unified_id': best_match['unified_person_id'],
                    'ps_name': best_match['person_name'],
                    'ps_harmonized': best_match['harmonized_name'],
                    'match_score': best_score,
                    'match_type': '2gram',
                    'cb_job_title': cb_row.get('job_title', ''),
                    'cb_organization': cb_row.get('featured_job_organization_name', '')
                })
        
        return matches
    
    def perform_people_matching(self, cb_people: pd.DataFrame, ps_inventors: pd.DataFrame) -> pd.DataFrame:
        """Perform people matching using 2-gram similarity"""
        logger.info("Matching people using 2-gram similarity...")
        
        # Filter CB people by job title (exclude finance, marketing, sales)
        excluded_titles = ['finance', 'marketing', 'sales', 'hr', 'legal', 'accounting', 'administration']
        if 'job_title' in cb_people.columns:
            mask = ~cb_people['job_title'].str.lower().str.contains('|'.join(excluded_titles), na=False)
            cb_people_filtered = cb_people[mask].copy()
            logger.info(f"Filtered people: {len(cb_people)} → {len(cb_people_filtered)}")
        else:
            cb_people_filtered = cb_people.copy()
        
        # Add harmonized names
        cb_people_filtered['harmonized_name'] = self.harmonize_names_parallel(cb_people_filtered['name'], 'person')
        
        # Group inventors by unified ID
        inventor_groups = ps_inventors.groupby('unified_person_id').agg({
            'person_id': 'first',
            'person_name': 'first',
            'harmonized_name': 'first',
            'person_ctry_code': lambda x: x.mode()[0] if len(x.mode()) > 0 else ''
        }).reset_index()
        
        logger.info(f"Matching {len(cb_people_filtered)} CB people with {len(inventor_groups)} unique PS inventors")
        
        # Split into chunks - use pandas-friendly approach
        num_chunks = min(self.num_processes * 4, len(cb_people_filtered))
        chunk_size = max(1, len(cb_people_filtered) // num_chunks + 1)
        chunks = [cb_people_filtered.iloc[i:i + chunk_size] for i in range(0, len(cb_people_filtered), chunk_size)]
        chunk_data = [(chunk, inventor_groups) for chunk in chunks if len(chunk) > 0]
        
        logger.info(f"Processing {len(cb_people_filtered):,} people in {len(chunk_data)} chunks...")
        
        # Process chunks in parallel with progress bar
        with Pool(self.num_processes) as pool:
            with tqdm(total=len(chunk_data), desc="People 2-gram matching") as pbar:
                chunk_results = []
                for result in pool.imap(self.compute_people_matches_chunk, chunk_data):
                    chunk_results.append(result)
                    pbar.update(1)
        
        # Flatten results
        all_matches = [match for chunk in chunk_results for match in chunk]
        logger.info(f"Found {len(all_matches)} people matches")
        
        return pd.DataFrame(all_matches)
    
    def get_company_inventors_batch(self, applicants_df: pd.DataFrame, inventors_df: pd.DataFrame) -> Dict:
        """Get inventors for each company applicant - batch optimized"""
        logger.info("Building company-inventor mapping...")
        
        # Merge applicants with inventors on appln_id
        company_inventors = applicants_df[['person_id', 'appln_id']].merge(
            inventors_df[['appln_id', 'person_name', 'person_ctry_code']].rename(
                columns={'person_name': 'inventor_name', 'person_ctry_code': 'inventor_country'}
            ),
            on='appln_id'
        )
        
        # Group by company person_id
        grouped = company_inventors.groupby('person_id').apply(
            lambda x: [{'name': n, 'country': c} for n, c in zip(x['inventor_name'], x['inventor_country'])]
        ).to_dict()
        
        return grouped
    
    def run(self, org_csv_path: str = None, people_csv_path: str = None,
            cleantech_appln_ids: Optional[pd.DataFrame] = None,
            cleantech_person_ids: Optional[pd.DataFrame] = None):
        """Run the complete matching process"""
        start_time = time.time()
        
        try:
            # Overall progress tracking
            total_milestones = 7
            current_milestone = 0
            
            # Extract data
            current_milestone += 1
            logger.info("="*60)
            logger.info(f"MILESTONE {current_milestone}/{total_milestones}: EXTRACTING DATA")
            logger.info("="*60)
            
            # In HPC mode, skip PATSTAT extraction and look for checkpoint
            if self.hpc_mode:
                logger.info("🔄 HPC mode: Looking for PATSTAT data checkpoint...")
                patstat_checkpoint = self.load_checkpoint("01_raw_patstat_data")
                if patstat_checkpoint is None:
                    raise FileNotFoundError(
                        f"PATSTAT data checkpoint not found in {self.checkpoint_dir}. "
                        "Please run the local extraction first and transfer the checkpoint files."
                    )
                patstat_data = patstat_checkpoint
                logger.info("✅ Loaded PATSTAT data from checkpoint")
            else:
                # Local mode: extract from PostgreSQL
                if cleantech_appln_ids is not None:
                    logger.info("🔍 Loading cleantech filters...")
                    cleantech_appln_path = os.path.join(os.path.dirname(self.checkpoint_dir), 'edv_tek_all_cleantech_appln_ids.csv')
                    cleantech_person_path = os.path.join(os.path.dirname(self.checkpoint_dir), 'edv_tek_all_cleantech_patstat_person_id_deduplicated.csv')
                    
                    if os.path.exists(cleantech_appln_path):
                        cleantech_appln_ids = pd.read_csv(cleantech_appln_path)
                        logger.info(f"   Loaded {len(cleantech_appln_ids):,} cleantech application IDs")
                    
                    if os.path.exists(cleantech_person_path):
                        cleantech_person_ids = pd.read_csv(cleantech_person_path)
                        logger.info(f"   Loaded {len(cleantech_person_ids):,} cleantech person IDs")
                
                patstat_data = self.extract_patstat_data(cleantech_appln_ids, cleantech_person_ids)
            
            cb_companies, cb_people = self.extract_crunchbase_data(org_csv_path, people_csv_path)
            
            # Exclude venture capital firms
            if 'roles' in cb_companies.columns:
                cb_companies = cb_companies[~cb_companies['roles'].str.contains('investor', na=False)]
                logger.info(f"Excluded VC firms, remaining organizations: {len(cb_companies)}")
            
            # 🏁 MILESTONE 3: Check if country codes already refilled
            milestone_name = "03_country_codes_refilled"
            checkpoint = self.load_checkpoint(milestone_name)
            if checkpoint is not None:
                logger.info("✅ Using cached data with refilled country codes")
                patstat_data = checkpoint['patstat_data']
                cb_companies = checkpoint['cb_companies']
                cb_people = checkpoint['cb_people']
            else:
                # Refill country codes
                current_milestone += 1
                logger.info(f"🔄 MILESTONE {current_milestone}/{total_milestones}: Refilling country codes...")
                logger.info("="*60)
                logger.info("REFILLING COUNTRY CODES")
                logger.info("="*60)
                
                patstat_data['applicants'] = self.refill_country_codes_batch(patstat_data['applicants'], 'patstat')
                patstat_data['inventors'] = self.refill_country_codes_batch(patstat_data['inventors'], 'patstat')
                cb_companies = self.refill_country_codes_batch(cb_companies, 'crunchbase')
                cb_people = self.refill_country_codes_batch(cb_people, 'crunchbase')
                
                # 💾 MILESTONE 3: Save data with refilled country codes
                country_data = {
                    'patstat_data': patstat_data,
                    'cb_companies': cb_companies,
                    'cb_people': cb_people
                }
                self.save_checkpoint(
                    "03_country_codes_refilled",
                    country_data,
                    "Data with refilled country codes using homonym analysis"
                )
            
            # 🏁 MILESTONE 4: Check if inventors already disambiguated
            milestone_name = "04_inventors_disambiguated"
            checkpoint = self.load_checkpoint(milestone_name)
            if checkpoint is not None:
                logger.info("✅ Using cached disambiguated inventors data")
                patstat_data = checkpoint['patstat_data']
                company_inventors = checkpoint['company_inventors']
            else:
                # Disambiguate inventors
                current_milestone += 1
                logger.info(f"🔄 MILESTONE {current_milestone}/{total_milestones}: Disambiguating inventors...")
                logger.info("="*60)
                logger.info("DISAMBIGUATING INVENTORS")
                logger.info("="*60)
                
                patstat_data['inventors'] = self.disambiguate_inventors_optimized(
                    patstat_data['inventors'], 
                    patstat_data['ipc']
                )
                
                # Get company-inventor mapping
                company_inventors = self.get_company_inventors_batch(
                    patstat_data['applicants'], 
                    patstat_data['inventors']
                )
                
                # 💾 MILESTONE 4: Save disambiguated data
                disambig_data = {
                    'patstat_data': patstat_data,
                    'company_inventors': company_inventors
                }
                self.save_checkpoint(
                    "04_inventors_disambiguated",
                    disambig_data,
                    "PATSTAT data with disambiguated inventors and company-inventor mapping"
                )
            
            # 🏁 MILESTONE 5: Check if company matching already done
            milestone_name = "05_company_matches"
            checkpoint = self.load_checkpoint(milestone_name)
            if checkpoint is not None:
                logger.info("✅ Using cached company matches")
                company_matches_df = checkpoint['company_matches']
            else:
                # Match companies
                current_milestone += 1
                logger.info(f"🔄 MILESTONE {current_milestone}/{total_milestones}: Matching companies...")
                logger.info("="*60)
                logger.info("MATCHING COMPANIES")
                logger.info("="*60)
                
                company_matches_df = self.perform_company_matching_staged(
                    cb_companies, 
                    patstat_data['applicants'],
                    company_inventors
                )
                
                # 💾 MILESTONE 5: Save company matches
                company_match_data = {
                    'company_matches': company_matches_df
                }
                self.save_checkpoint(
                    "05_company_matches",
                    company_match_data,
                    "Company matching results using 4-stage matching algorithm"
                )
            
            # 🏁 MILESTONE 6: Check if people matching already done
            milestone_name = "06_people_matches"
            checkpoint = self.load_checkpoint(milestone_name)
            if checkpoint is not None:
                logger.info("✅ Using cached people matches")
                people_matches_df = checkpoint['people_matches']
            else:
                # Match people
                current_milestone += 1
                logger.info(f"🔄 MILESTONE {current_milestone}/{total_milestones}: Matching people...")
                logger.info("="*60)
                logger.info("MATCHING PEOPLE")
                logger.info("="*60)
                
                people_matches_df = self.perform_people_matching(cb_people, patstat_data['inventors'])
                
                # 💾 MILESTONE 6: Save people matches
                people_match_data = {
                    'people_matches': people_matches_df
                }
                self.save_checkpoint(
                    "06_people_matches",
                    people_match_data,
                    "People matching results using 2-gram similarity algorithm"
                )
            
            # Save final results
            current_milestone += 1
            logger.info(f"🔄 MILESTONE {current_milestone}/{total_milestones}: Saving final results...")
            logger.info("="*60)
            logger.info("SAVING FINAL RESULTS")
            logger.info("="*60)
            
            # Set output path based on mode
            if self.hpc_mode:
                output_path = os.path.join(self.hpc_data_dir, 'matching_results/')
            else:
                output_path = '/mnt/hdd02/Projekt_EDV_TEK/matching_results/'
            os.makedirs(output_path, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save to CSV
            company_csv = f"{output_path}company_matches_{timestamp}.csv"
            people_csv = f"{output_path}people_matches_{timestamp}.csv"
            
            company_matches_df.to_csv(company_csv, index=False)
            people_matches_df.to_csv(people_csv, index=False)
            
            logger.info(f"📄 Company matches saved: {company_csv}")
            logger.info(f"📄 People matches saved: {people_csv}")
            
            # 💾 MILESTONE 7: Save final results checkpoint
            final_results = {
                'company_matches': company_matches_df,
                'people_matches': people_matches_df
            }
            self.save_checkpoint(
                "07_final_results",
                final_results,
                "Final matching results - companies and people"
            )
            
            # Generate statistics
            elapsed_time = time.time() - start_time
            stats = {
                'execution_date': datetime.now().isoformat(),
                'execution_time_seconds': elapsed_time,
                'execution_time_hours': elapsed_time / 3600,
                'input_data': {
                    'crunchbase_organizations': len(cb_companies),
                    'crunchbase_people': len(cb_people),
                    'patstat_applicants': len(patstat_data['applicants']),
                    'patstat_inventors': len(patstat_data['inventors']),
                    'patstat_unique_inventors': patstat_data['inventors']['unified_person_id'].nunique()
                },
                'company_matches': {
                    'total': len(company_matches_df),
                    'by_type': company_matches_df['match_type'].value_counts().to_dict() if len(company_matches_df) > 0 else {},
                    'by_stage': company_matches_df['match_stage'].value_counts().to_dict() if len(company_matches_df) > 0 else {},
                },
                'people_matches': {
                    'total': len(people_matches_df),
                    'average_score': people_matches_df['match_score'].mean() if len(people_matches_df) > 0 else 0
                }
            }
            
            stats_file = f"{output_path}matching_statistics_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"📊 Statistics saved: {stats_file}")
            
            logger.info("="*60)
            logger.info("MATCHING SUMMARY")
            logger.info("="*60)
            logger.info(f"Execution time: {elapsed_time/60:.2f} minutes")
            logger.info(f"Company matches: {len(company_matches_df)}")
            logger.info(f"People matches: {len(people_matches_df)}")
            
            return company_matches_df, people_matches_df
            
        except Exception as e:
            logger.error(f"Error during matching: {str(e)}", exc_info=True)
            raise


def main():
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PATSTAT-Crunchbase Matcher with Checkpoints')
    parser.add_argument('--organizations-csv', help='Path to Crunchbase organizations CSV file (optional in HPC mode)')
    parser.add_argument('--people-csv', help='Path to Crunchbase people CSV file (optional in HPC mode)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from existing checkpoints if available')
    parser.add_argument('--clear-checkpoints', action='store_true',
                       help='Clear all existing checkpoints before starting')
    parser.add_argument('--checkpoint-dir', default='/mnt/hdd02/Projekt_EDV_TEK/matching_checkpoints',
                       help='Directory to store checkpoints (default: /mnt/hdd02/Projekt_EDV_TEK/matching_checkpoints)')
    parser.add_argument('--hpc-mode', action='store_true',
                       help='Run in HPC mode (skip PostgreSQL operations, auto-detect files)')
    parser.add_argument('--hpc-data-dir', default='/fibus/fs1/0f/cyh1826/wt/edv_tek/',
                       help='HPC data directory (default: /fibus/fs1/0f/cyh1826/wt/edv_tek/)')
    parser.add_argument('--local-only', action='store_true',
                       help='Run only local PostgreSQL extraction and exit')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if not args.hpc_mode and not args.local_only:
        if not args.organizations_csv or not args.people_csv:
            print("❌ Error: In local mode, both --organizations-csv and --people-csv are required")
            parser.print_help()
            sys.exit(1)
    
    org_csv_path = args.organizations_csv
    people_csv_path = args.people_csv
    
    # Clear checkpoints if requested
    if args.clear_checkpoints:
        import shutil
        if os.path.exists(args.checkpoint_dir):
            shutil.rmtree(args.checkpoint_dir)
            print(f"🗑️ Cleared all checkpoints from: {args.checkpoint_dir}")
    
    print(f"📁 Using checkpoint directory: {args.checkpoint_dir}")
    if args.resume:
        print("🔄 Resume mode: Will use existing checkpoints if available")
    
    # Database configuration for PATSTAT only
    patstat_config = {
        'host': '',
        'port': '',
        'database': '',
        'user': '',
        'password': ''
    }
    
    # Load cleantech filters - CRITICAL for performance
    cleantech_appln_ids = None
    cleantech_person_ids = None
    
    # Set paths based on mode
    if args.hpc_mode:
        cleantech_patents_path = os.path.join(args.hpc_data_dir, 'edv_tek_all_cleantech_appln_ids.csv')
        cleantech_persons_path = os.path.join(args.hpc_data_dir, 'edv_tek_all_cleantech_patstat_person_id_deduplicated.csv')
    else:
        cleantech_patents_path = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_all_cleantech_appln_ids.csv'
        cleantech_persons_path = '/mnt/hdd02/Projekt_EDV_TEK/edv_tek_all_cleantech_patstat_person_id_deduplicated.csv'
    
    if os.path.exists(cleantech_patents_path):
        cleantech_appln_ids = pd.read_csv(cleantech_patents_path)
        logger.info(f"Loaded {len(cleantech_appln_ids)} cleantech application IDs")
    else:
        logger.warning(f"Cleantech patents file not found at: {cleantech_patents_path}")
        logger.warning("Running without cleantech filter will process ALL patents (very slow!)")
    
    if os.path.exists(cleantech_persons_path):
        cleantech_person_ids = pd.read_csv(cleantech_persons_path)
        logger.info(f"Loaded {len(cleantech_person_ids)} cleantech person IDs")
    else:
        logger.warning(f"Cleantech persons file not found at: {cleantech_persons_path}")
        logger.warning("Running without cleantech filter will process ALL persons (very slow!)")
    
    # Initialize matcher
    matcher = OptimizedCrunchbasePATSTATMatcher(
        patstat_config=patstat_config if not args.hpc_mode else None,
        num_processes=None,  # Will auto-detect: 32 for HPC, 24 for local
        chunk_size=10000,
        checkpoint_dir=args.checkpoint_dir,
        hpc_mode=args.hpc_mode,
        hpc_data_dir=args.hpc_data_dir
    )
    
    # Handle different execution modes
    if args.local_only:
        logger.info("🏠 LOCAL-ONLY MODE: Running PostgreSQL extraction only...")
        patstat_data = matcher.extract_patstat_data(cleantech_appln_ids, cleantech_person_ids)
        logger.info("✅ Local extraction complete. Transfer checkpoint files to HPC environment.")
        logger.info(f"📁 Checkpoint files location: {matcher.checkpoint_dir}")
        return
    
    # Run full matching process
    company_results, people_results = matcher.run(org_csv_path, people_csv_path, cleantech_appln_ids, cleantech_person_ids)
    
    print(f"\nMatching completed!")
    print(f"Total company matches: {len(company_results)}")
    print(f"Total people matches: {len(people_results)}")


if __name__ == "__main__":
    main() 