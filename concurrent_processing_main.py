import os
import datetime
import time
import json
import pandas as pd
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import multiprocessing

from service.fetchers.srp_fetcher import fetch_wizqa
from service.utils.config import get_logger, setup_file_logging, get_log_file_path
from service.fetch_pipeline import run_pipeline
from service.evaluations.sa_blocklist import scan_json_files
from service.evaluations.death_check import run_death_check
from service.evaluations.sa_relevance import check_relevance_pair
from service.evaluations.kg_relevance import run_kg_mismatch_check

log_file = setup_file_logging("logs")
logger = get_logger()

# Config
WIZQA_DIR = "data/wizqa"
BLOCKLIST_PATH = "resource/sa_blocklist.json"

# Concurrent processing configuration
MAX_CONCURRENT_WIZQA = 10  # We can limit concurrent WIZQA fetches to 10 to avoid API overload
MAX_CONCURRENT_FILES = 20  # We can limit concurrent file processing to 20 to avoid memory issues
USE_CONCURRENT = True  # Toggle to easily switch between concurrent and sequential


class ConcurrentProcessor:
    """Handles concurrent processing with proper resource management."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=min(32, (multiprocessing.cpu_count() or 1) * 4)
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


def run_trend_pipeline():
    """Run the full SQA trend pipeline and return all issues found."""
    logger.info("📈 [1/6] Running trend pipeline (with context)")
    trend_data = run_pipeline()
    queries = list(trend_data.keys())
    logger.info(f"Retrieved {len(queries)} queries")
    return trend_data, queries


async def fetch_wizqa_batch_async(queries_batch: List[str], semaphore: asyncio.Semaphore):
    """Fetch WIZQA for a batch of queries with rate limiting."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, fetch_wizqa, queries_batch)


async def fetch_and_save_wizqa_concurrent(queries: List[str]) -> str:
    """Fetch WIZQA responses for all queries concurrently."""
    logger.info(f"🧠 [2/6] Fetching WIZQA responses (concurrent mode for {len(queries)} queries)")
    
    # Splitting queries into batches to avoid overwhelming the API
    batch_size = 5
    query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    # Creating semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WIZQA)

    # Creating tasks for all batches
    tasks = [
        fetch_wizqa_batch_async(batch, semaphore) 
        for batch in query_batches
    ]
    
    await asyncio.gather(*tasks)
    
    latest_wizqa_folder = sorted(os.listdir(WIZQA_DIR))[-1]
    wizqa_path = os.path.join(WIZQA_DIR, latest_wizqa_folder)
    return wizqa_path


def fetch_and_save_wizqa(queries):
    """Fetch WIZQA responses for all queries (with optional concurrent mode)."""
    if USE_CONCURRENT and len(queries) > 5:  # Use concurrent for more than 5 queries
        return asyncio.run(fetch_and_save_wizqa_concurrent(queries))
    else:
        logger.info("🧠 [2/6] Fetching WIZQA responses")
        fetch_wizqa(queries)
        latest_wizqa_folder = sorted(os.listdir(WIZQA_DIR))[-1]
        wizqa_path = os.path.join(WIZQA_DIR, latest_wizqa_folder)
        return wizqa_path


def scan_blocklist(wizqa_path):
    """Scan WIZQA results for blocklist matches."""
    logger.info("🧹 [3/6] Running SA blocklist scan")
    sa_results = scan_json_files(wizqa_path, BLOCKLIST_PATH)
    logger.info(f"🔍 Blocklist matches found: {len(sa_results)}")
    for r in sa_results:
        logger.info(f"⚠️  Query: {r['query']}, Category: {r['category']}, "
                    f"Match: '{r['matched_token']}' in module {r['module']}")
    return sa_results


def scan_death_context(trend_data):
    """Scan for death-related context in trend data."""
    logger.info("💀 [4/6] Running death context check")
    death_results = run_death_check(trend_data)
    logger.info(f"☠️  Death-related context matches found: {len(death_results)}")
    return death_results


async def process_relevance_file_async(filepath: str, executor: ThreadPoolExecutor) -> List[Dict]:
    """Process a single file for relevance check asynchronously."""
    results = []
    filename = os.path.basename(filepath)
    
    try:
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
        
        query = data.get("data", {}).get("search", {}).get("query", "").strip()
        paa = (
            data.get("data", {})
                .get("search", {})
                .get("data", {})
                .get("peopleAlsoAsk", {})
                .get("peopleAlsoAsk", {})
                .get("data", {})
                .get("list", [])
        )
        suggestions = [item.get("title", "") for item in paa if "title" in item]
        
        loop = asyncio.get_event_loop()
        for suggestion in suggestions:
            is_irrelevant = await loop.run_in_executor(
                executor,
                check_relevance_pair,
                query,
                suggestion
            )
            if is_irrelevant:
                results.append({
                    "query": query,
                    "module": "peopleAlsoAsk",
                    "offending_string": suggestion,
                    "matched_token": "llm_irrelevant",
                    "category": "off_topic",
                    "error_type": "relevance",
                    "is_dead": "no"
                })
                
        if results:
            logger.info(f"[🔍] Found {len(results)} relevance issues in {filename}")
            
    except Exception as e:
        logger.warning(f"⚠️ Error processing {filename}: {e}")
    
    return results


async def check_relevance_concurrent(wizqa_path: str) -> List[Dict]:
    """Run relevance check on WIZQA suggestions concurrently."""
    logger.info("🧠 [5/6] Running relevance check on suggestions (concurrent)")
    
    json_files = [
        os.path.join(wizqa_path, f) 
        for f in os.listdir(wizqa_path) 
        if f.endswith(".json")
    ]
    
    relevance_results = []
    
    # Processing files concurrently with semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILES)
    
    with ConcurrentProcessor() as processor:
        async def process_with_semaphore(filepath):
            async with semaphore:
                return await process_relevance_file_async(filepath, processor.executor)
        
        tasks = [process_with_semaphore(f) for f in json_files]
        results_lists = await asyncio.gather(*tasks)
        
        relevance_results = [item for sublist in results_lists for item in sublist]
    
    logger.info(f"🔍 Relevance mismatches found: {len(relevance_results)}")
    return relevance_results


def check_relevance(wizqa_path):
    """Run relevance check on WIZQA suggestions."""
    if USE_CONCURRENT:
        return asyncio.run(check_relevance_concurrent(wizqa_path))
    else:
        logger.info("🧠 [5/6] Running relevance check on suggestions")
        relevance_results = []
        for filename in os.listdir(wizqa_path):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(wizqa_path, filename)
            logger.info(f"[🔍] Processing file: {filename}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                query = data.get("data", {}).get("search", {}).get("query", "").strip()
                paa = (
                    data.get("data", {})
                        .get("search", {})
                        .get("data", {})
                        .get("peopleAlsoAsk", {})
                        .get("peopleAlsoAsk", {})
                        .get("data", {})
                        .get("list", [])
                )
                suggestions = [item.get("title", "") for item in paa if "title" in item]
                for suggestion in suggestions:
                    if check_relevance_pair(query, suggestion):
                        relevance_results.append({
                            "query": query,
                            "module": "peopleAlsoAsk",
                            "offending_string": suggestion,
                            "matched_token": "llm_irrelevant",
                            "category": "off_topic",
                            "error_type": "relevance",
                            "is_dead": "no"
                        })
            except Exception as e:
                logger.warning(f"⚠️ Error processing {filename}: {e}")
        logger.info(f"🔍 Relevance mismatches found: {len(relevance_results)}")
        return relevance_results


def check_kg_mismatch(wizqa_path):
    """Run KG People match check."""
    logger.info("👤 [6/6] Running KG People match check")
    kg_mismatch_results = run_kg_mismatch_check(wizqa_path)
    logger.info(f"🔍 KG mismatches found: {len(kg_mismatch_results)}")
    return kg_mismatch_results


def save_issues_report(all_issues, latest_wizqa_folder):
    """Save all issues to a CSV report."""
    if all_issues:
        df = pd.DataFrame(all_issues)
        column_order = [
            "error_type", "query", "module", "offending_string",
            "matched_token", "category", "is_dead"
        ]
        df = df[[col for col in column_order if col in df.columns]]
        csv_name = f"{latest_wizqa_folder}.csv"
        output_path = os.path.join("reports", csv_name)
        os.makedirs("reports", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"📄 Combined results saved to: {output_path}")
    else:
        logger.info("✅ No issues found in blocklist, death check, relevance, or KG mismatch.")


def combine_all_issues(sa_results, death_results, relevance_results, kg_mismatch_results):
    """Combine all issue lists into a single list with proper annotations."""
    all_issues = []
    for r in sa_results:
        r["error_type"] = "SA blocklist"
        r["is_dead"] = "no"
        all_issues.append(r)
    all_issues.extend(death_results)
    all_issues.extend(relevance_results)
    all_issues.extend(kg_mismatch_results)
    return all_issues


async def run_evaluations_concurrent(wizqa_path: str, trend_data: Dict) -> Tuple:
    """Run all evaluation steps concurrently."""
    logger.info("🚀 Running evaluations in parallel...")
    
    loop = asyncio.get_event_loop()
    
    with ConcurrentProcessor() as processor:
        sa_task = loop.run_in_executor(
            processor.executor, scan_blocklist, wizqa_path
        )
        death_task = loop.run_in_executor(
            processor.executor, scan_death_context, trend_data
        )
        kg_task = loop.run_in_executor(
            processor.executor, check_kg_mismatch, wizqa_path
        )
        
        relevance_task = check_relevance_concurrent(wizqa_path)
        
        sa_results, death_results, kg_results = await asyncio.gather(
            sa_task, death_task, kg_task
        )
        relevance_results = await relevance_task
    
    return sa_results, death_results, relevance_results, kg_results


def process_concurrent():
    """Concurrent version of the main process function."""
    # Step 1: Generate trends (synchronous)
    trend_data, queries = run_trend_pipeline()
    
    # Step 2: Fetch WIZQA responses (concurrent)
    wizqa_path = fetch_and_save_wizqa(queries)
    latest_wizqa_folder = os.path.basename(wizqa_path)
    
    # Steps 3-6: Run all evaluations concurrently
    if USE_CONCURRENT:
        sa_results, death_results, relevance_results, kg_mismatch_results = asyncio.run(
            run_evaluations_concurrent(wizqa_path, trend_data)
        )
    else:
        # Sequential fallback
        sa_results = scan_blocklist(wizqa_path)
        death_results = scan_death_context(trend_data)
        relevance_results = check_relevance(wizqa_path)
        kg_mismatch_results = check_kg_mismatch(wizqa_path)
    
    # Combine and save results
    all_issues = combine_all_issues(
        sa_results, death_results, relevance_results, kg_mismatch_results
    )
    save_issues_report(all_issues, latest_wizqa_folder)
    
    return all_issues


def process():
    """
    1. Generate trending queries.
    2. Fetch WIZQA responses for each query.
    3. Scan results for blocklist matches.
    4. Check for death-related context.
    5. Evaluate relevance of suggestions.
    6. Check for KG People mismatches.
    7. Combine and save all issues to a report.
    """
    if USE_CONCURRENT:
        return process_concurrent()
    else:
        trend_data, queries = run_trend_pipeline()
        wizqa_path = fetch_and_save_wizqa(queries)
        latest_wizqa_folder = os.path.basename(wizqa_path)
        sa_results = scan_blocklist(wizqa_path)
        death_results = scan_death_context(trend_data)
        relevance_results = check_relevance(wizqa_path)
        kg_mismatch_results = check_kg_mismatch(wizqa_path)
        all_issues = combine_all_issues(sa_results, death_results, relevance_results, kg_mismatch_results)
        save_issues_report(all_issues, latest_wizqa_folder)
        return all_issues


if __name__ == "__main__":
    dt = datetime.datetime.now().isoformat()
    logger.info(f"=== New run started at {dt} ===")
    logger.info(f"🚀 Concurrent processing: {'ENABLED' if USE_CONCURRENT else 'DISABLED'}")
    
    start_time = time.time()
    result = process()
    end_time = time.time()
    
    elapsed = end_time - start_time
    logger.info(f"=== Run finished in {elapsed:.2f} seconds ===")
    
    if USE_CONCURRENT:
        estimated_sequential_time = elapsed * 4
        logger.info(f"⚡ Estimated time saved: {estimated_sequential_time - elapsed:.2f} seconds")