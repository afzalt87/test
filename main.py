# package imports
import os
import logging
import datetime
import time
import yaml
import json

# service imports
from service.llm import Llm
from service.output_formatter import format_report
from service.send_email import send_email
from service.evaluations.blocklist import load_all_blocklists, detect_blocklist
from service.fetchers.srp_fetcher import fetch_sayt, fetch_png, fetch_html, fetch_wizqa
from service.fetchers.trend_fetcher import generate_trends
from service.logger_setup import setup_logging
from service.filter_resource import filter_srp_resources
from service.read_txt_file import read_txt_file


# logger setup
LOG_FILE = setup_logging("logs")
logger = logging.getLogger(__name__)


def text_check(query):
    # TODO implement text check logic, https://git.ouryahoo.com/CAKE/CAKE-AI/tree/main/QA_Automation/keyword_detection_poc
    # each key of the output JSON is an evaluation, for example {"blocklist": blocklist_result, "relevance": relevance_result}

    html_path = fetch_html(query)
    html_content = None
    if not html_path:
        logger.warning(f"No HTML file fetched for query: {query}")
    else:
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            logger.error(
                f"Failed to read HTML file {html_path} for query {query}: {e}")
    filtered_content = filter_srp_resources("html", html_content)
    if not filtered_content:
        logger.warning(f"Filtered HTML content is empty for query: {query}")
    list_to_check = load_all_blocklists()
    blocklist_result = detect_blocklist(filtered_content, list_to_check)

    return {"blocklist": blocklist_result}


def image_check(query):
    # TODO [Justine] extract image check logic from https://git.ouryahoo.com/CAKE-AI/Trending-Searches-Module-Analysis/blob/afzalt-patch-1/trending-searches-module-analysis-using-gpt.ipynb?short_path=a600d59, # each key of the output JSON is an evaluation, for example {"image_crop": llm_result}

    img_path = fetch_png(query)
    if not img_path:
        logger.warning(f"No image file fetched for query: {query}")
    system_prompt = read_txt_file("resource/prompt/system/example.txt")
    user_prompt = read_txt_file("resource/prompt/user/example.txt")
    llm_result = Llm().call_with_image(system_prompt, user_prompt, img_path)
    if llm_result is None:
        logger.warning(f"LLM image check returned None for query: {query}")
    # TODO [Justine] maybe implement a format check for llm_result
    return {"image_check": llm_result}


def run_process():
    trends = generate_trends()
    results = []
    for query in trends:
        logger.info(f"Processing query: {query}")
        text_check_result = text_check(query)
        image_check_result = image_check(query)
        combined_result = {**text_check_result, **image_check_result}
        results.append({"query": query, "report": combined_result})

    return results


if __name__ == "__main__":
    dt = datetime.datetime.now().isoformat()
    logger.info(f"=== New run started at {dt} ===")
    start_time = time.time()
    result = run_process()
    report = format_report(result)
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"=== Run finished in {elapsed:.2f} seconds ===")
    recipients = os.getenv("TO_EMAIL")
    try:
        send_email(f"[SQA Report] {dt}", report, recipients, LOG_FILE)
        logger.info("Report email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send report email: {e}")
