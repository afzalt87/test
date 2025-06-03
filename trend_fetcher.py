import logging

logger = logging.getLogger(__name__)


def generate_trends():
    try:
        # TODO [Kent]: Implement the actual trend generation logic, from https://git.ouryahoo.com/CAKE/CAKE-AI/tree/main/QA_Automation/20250512_get_trends_01, without the slack bot process and the token tracking.

        # sould return a list of queries
        return ["harvard", "computex"]
    except Exception as e:
        logger.exception(f"Trend generation error: {e}")
