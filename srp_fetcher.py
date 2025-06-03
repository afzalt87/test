import logging
import requests
import yaml
import json
import time

logger = logging.getLogger(__name__)
with open("env_settings.yaml", "r") as f:
    env_settings = yaml.safe_load(f)


def _fetch_screenshot_resource(query, format, output_dir, ext):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        }
        query_encode = requests.utils.quote(query)
        screenshot_n_cache_api = env_settings.get("SCREENSHOT_N_CACHE_API")
        yahoo_search = env_settings.get("YAHOO_US_SRP")
        response = requests.post(
            f"{screenshot_n_cache_api}/capture_page",
            params={
                "target_url": f'{yahoo_search}?p={query_encode}',
                "format": format,
                "device": "desktop"
            },
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data.get("task_id")
        if not task_id:
            logger.error(
                f"No task_id returned for query '{query}' (format: {format})")
            return None

        status = None
        max_retries = 30
        retries = 0
        while status not in {"completed", "failed"} and retries < max_retries:
            status_response = requests.get(
                f"{screenshot_n_cache_api}/status/{task_id}", timeout=5)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("status")
            logger.info(
                f"{format.upper()} task {task_id} status: {status} (query: '{query}')")
            if status not in {"completed", "failed"}:
                time.sleep(1)
                retries += 1
        if status == "completed":
            output_filename = f"{output_dir}/{task_id}.{ext}"
            result_response = requests.get(
                f"{screenshot_n_cache_api}/result/{task_id}", timeout=10)
            result_response.raise_for_status()
            with open(output_filename, "wb") as f:
                f.write(result_response.content)
            logger.info(
                f"{format.upper()} saved to {output_filename} for query '{query}'")
            return output_filename
        else:
            logger.error(
                f"{format.upper()} task {task_id} failed or timed out for query '{query}'")
            return None
    except Exception as e:
        logger.exception(f"Error fetching SRP {format} for '{query}': {e}")
        return None


def fetch_png(query):
    return _fetch_screenshot_resource(query, format="png", output_dir="static/img", ext="png")


def fetch_html(query):
    return _fetch_screenshot_resource(query, format="html", output_dir="static/html", ext="html")


def fetch_wizqa(query):
    """
    Fetch a text representation of the SRP for the given query.
    Returns: dict (currently returning example JSON content from redux_api_example.json)
    """
    try:
        # TODO [Adem]: implement redux api logic, please also adjust the function name, and drop def fetch_sayt(query) + remove example json file
        with open("redux_api_example.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.exception(f"Error fetching SRP text for '{query}': {e}")
        return None


def fetch_sayt(query):
    """
    Fetch a search-as-you-type (SAYT) suggestion for the given query using the SAYT_API from env_settings.yaml.
    Returns: dict with key 'data' containing a list of suggestion strings.
    """
    try:
        sayt_api = env_settings.get("SAYT_API")
        if not sayt_api:
            logger.error("SAYT_API not found in env_settings.yaml")
            return None
        params = {"command": query, "nresults": 10, "output": "yjson"}
        response = requests.get(sayt_api, params=params)
        response.raise_for_status()
        data = response.json()
        # Extract only the suggestion strings from the 'r' field of api output
        suggestions = [item[0] for item in data.get("r", [])]
        return {"data": suggestions}
    except Exception as e:
        logger.exception(f"Error fetching SAYT for query '{query}': {e}")
        return None
