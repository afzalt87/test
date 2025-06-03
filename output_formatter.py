import logging
import json

logger = logging.getLogger(__name__)

# TODO attach html into report?
# TODO in report specify where is the issue


def format_report(raw_results):
    # TODO: clean up the result list and return an ouput that is easy to read, function currently does basically nothing
    return json.dumps(raw_results, indent=2, ensure_ascii=False)
