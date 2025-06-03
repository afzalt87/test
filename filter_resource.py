import logging
import requests
import yaml
import json
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
with open("env_settings.yaml", "r") as f:
    env_settings = yaml.safe_load(f)


def filter_srp_resources(type, srp_resources, specific_keys=[]):
    # TODO: Allow filtering of srp_resources, to only keep what is needed. This method has fake logic
    try:
        if type == "html":
            soup = BeautifulSoup(srp_resources, "html.parser")
            h3_tags = [h3.get_text(strip=True) for h3 in soup.find_all("h3")]
            # Join all h3 tag texts into a single string, separated by newlines
            return "\n".join(h3_tags)

        filtered_source = srp_resources
        return filtered_source
    except Exception as e:
        logger.exception(f"Error filtering SRP resources: '{e}'")
        return None
