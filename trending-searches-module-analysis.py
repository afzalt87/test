#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install --upgrade openai selenium seletools selenium-webdriver-extender webdriver-manager pillow tqdm chromedriver undetected-chromedriver')


# In[1]:


import os
import time
import json
import base64
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
from tqdm.notebook import tqdm
from io import BytesIO
from PIL import Image
import requests

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# OpenAI API
from openai import OpenAI

@dataclass
class TrendingItem:
    """Class for representing a trending item from Yahoo SERP"""
    position: int
    title: str
    description: Optional[str]
    image_url: str
    image_data: Optional[bytes] = None
    analysis_result: Optional[Dict] = None

class YahooTrendAnalyzerSelenium:
    """Class for analyzing Yahoo trending module images and titles using Selenium with web search capabilities"""
    
    def __init__(self, headless=True, timeout=30, api_key=None):
        """Initialize the analyzer
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Timeout for page navigation in seconds
            api_key: OpenAI API key. If None, tries to use environment variable
        """
        self.headless = headless
        self.timeout = timeout
        self.base_url = "https://search.yahoo.com/search?p={query}&fr=fp-tts&fr2=p:fp,m:tn,ct:foryou,kt:org,pg:1,stl:txt,b:"
        self.items = []
        self.driver = None
        
        # Setting up OpenAI client
        if api_key:
            self.openai_client = OpenAI(api_key=None)
        else:
            api_key_env = os.environ.get("OPENAI_API_KEY")
            if not api_key_env:
                print("WARNING: No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
                print("Image analysis functionality will be disabled.")
                self.openai_client = None
            else:
                self.openai_client = OpenAI(api_key=api_key_env)
    
    def initialize_browser(self):
        """Initialize Selenium browser"""
        print("Initializing Chrome browser...")
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            self.driver.set_page_load_timeout(self.timeout)
            return True
        except Exception as e:
            print(f"Error initializing browser: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def web_search(self, query: str) -> str:
        """Perform a web search and return results
        
        This is a mock implementation. In a real scenario, you would integrate
        with an actual search API like Bing, Google Custom Search, or SerpAPI.
        """
        print(f"Performing web search for: {query}")
        
        return f"Search results for '{query}': [This would contain actual search results from a search API]"
    
    def fetch_trending_module(self, query="news"):
        """Fetch the trending module from Yahoo SERP"""
        if not self.driver and not self.initialize_browser():
            print("Failed to initialize browser")
            return False
        
        url = self.base_url.format(query=quote_plus(query))
        
        try:
            print(f"Navigating to {url}...")
            self.driver.get(url)
            
            time.sleep(2)
            
            print("Looking for trending module...")
            
            # Trying various selectors to find the trending module
            selectors_to_try = [
                "div.dd.theme-default.refreshtn.wthThmb.trendingNow",
                ".trendingNow",
                "div.layoutMiddle ul li",
                "#right div.dd",
                "div[class*='trendingNow']",
                "div[class*='refreshtn']"
            ]
            
            module = None
            found_selector = None
            
            for selector in selectors_to_try:
                print(f"Trying selector: {selector}")
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        module = elements[0]
                        found_selector = selector
                        print(f"Found module with selector: {selector}")
                        break
                except Exception as e:
                    print(f"Error with selector {selector}: {e}")
            
            if not module:
                print("Taking screenshot for debugging...")
                self.driver.save_screenshot("yahoo_page.png")
                print("Saved screenshot as yahoo_page.png")
                print("Could not find trending module with any selector")
                return False
            
            print("Found trending module, extracting items...")
            
            trending_items = []
            
            item_selectors = [
                "li.story-item",
                "div.layoutMiddle ul li",
                "ul li"
            ]
            
            story_items = []
            for selector in item_selectors:
                try:
                    if found_selector:
                        full_selector = f"{found_selector} {selector}"
                        print(f"Trying to find items with: {full_selector}")
                        items = self.driver.find_elements(By.CSS_SELECTOR, full_selector)
                        if items:
                            story_items = items
                            print(f"Found {len(items)} items with selector: {full_selector}")
                            break
                    
                    # Trying global search as fallback
                    print(f"Trying global selector: {selector}")
                    items = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if items and len(items) > 2:  # We expect at least a few items
                        story_items = items
                        print(f"Found {len(items)} items with global selector: {selector}")
                        break
                except Exception as e:
                    print(f"Error with item selector {selector}: {e}")
            
            if not story_items:
                print("Could not find any trending items. Checking page source...")
                page_source = self.driver.page_source
                
                indicators = ["trendingNow", "trending searches", "Today's trending"]
                for indicator in indicators:
                    if indicator in page_source:
                        print(f"Found '{indicator}' in page source, but couldn't extract items")
                
                try:
                    any_items = self.driver.find_elements(By.CSS_SELECTOR, "#right li")
                    if any_items:
                        print(f"Found {len(any_items)} potential items as fallback")
                        story_items = any_items[:6]  # Just use the first few items
                except Exception:
                    pass
            
            for i, item in enumerate(story_items[:6]):  # Limit to 6 items
                try:
                    print(f"Processing item {i+1}...")
                    
                    try:
                        item_html = item.get_attribute('outerHTML')
                        print(f"Item HTML snippet: {item_html[:100]}...")
                    except:
                        print("Couldn't get item HTML")
                    
                    position = i + 1
                    title = "Unknown"
                    description = None
                    img_src = None
                    img_data = None
                    
                    title_selectors = ["h4 span", "h4", ".text-wrap h4", "span:not(.num)", ".title"]
                    for selector in title_selectors:
                        try:
                            title_elements = item.find_elements(By.CSS_SELECTOR, selector)
                            if title_elements:
                                title = title_elements[0].text
                                print(f"Found title: {title}")
                                break
                        except:
                            continue
                    
                    desc_selectors = ["p span", "p", ".text-wrap p"]
                    for selector in desc_selectors:
                        try:
                            desc_elements = item.find_elements(By.CSS_SELECTOR, selector)
                            if desc_elements:
                                description = desc_elements[0].text
                                print(f"Found description: {description}")
                                break
                        except:
                            continue
                    
                    img_selectors = ["img", "div > img", ".img-wrapper img"]
                    for selector in img_selectors:
                        try:
                            img_elements = item.find_elements(By.CSS_SELECTOR, selector)
                            if img_elements:
                                img_src = img_elements[0].get_attribute("src")
                                if not img_src or img_src.startswith("data:image/gif"):
                                    img_src = img_elements[0].get_attribute("data-src")
                                
                                if img_src:
                                    print(f"Found image URL: {img_src}")
                                    try:
                                        img_response = requests.get(img_src)
                                        if img_response.status_code == 200:
                                            img_data = img_response.content
                                        else:
                                            print(f"Failed to fetch image: HTTP {img_response.status_code}")
                                    except Exception as e:
                                        print(f"Error fetching image: {e}")
                                break
                        except:
                            continue
                    
                    item_data = TrendingItem(
                        position=position,
                        title=title if title != "Unknown" else f"Item {position}",
                        description=description,
                        image_url=img_src,
                        image_data=img_data
                    )
                    trending_items.append(item_data)
                    print(f"Added item {position}: {title}")
                
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.items = trending_items
            
            if not trending_items and module:
                print("Creating dummy items...")
                for i in range(1, 7):
                    self.items.append(TrendingItem(
                        position=i,
                        title=f"Trending Item {i}",
                        description="Could not extract content",
                        image_url="",
                        image_data=None
                    ))
            
            print(f"Successfully extracted {len(self.items)} items")
            return len(self.items) > 0
        
        except Exception as e:
            print(f"Error fetching trending module: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            pass
    
    
    def analyze_images(self, system_prompt=None, use_web_search=True):
        """Analyze images using GPT-4.1 with optional web search for context
        
        Args:
            system_prompt: Custom system prompt for the analysis
            use_web_search: Whether to enable web search tool for additional context
        """
        if not self.items:
            print("No items to analyze. Fetch trending module first.")
            return
        
        if not hasattr(self, 'openai_client') or self.openai_client is None:
            print("OpenAI API key not set. Skipping image analysis.")
            for item in self.items:
                item.analysis_result = {
                    "image_relevance": "N/A",
                    "text_relevance": "N/A", 
                    "image_quality": "N/A",
                    "image_integrity": "OpenAI API key not provided",
                    "justification": "Image analysis was skipped because no OpenAI API key was provided."
                }
            return
        
        if system_prompt is None:
            system_prompt = """You are an expert image analyst who specializes in image relevance and image
quality analysis. You have an eye for detail. When provided with a query, title, and
an image, you analyze the image thoroughly, do additional research if necessary to
understand the entity or subject, and then use your expertise to analyze the images
and provide accurate analysis.

You can use the web_search function to research topics, people, or events shown in the images
to provide more accurate and contextual analysis. Use web search when you need current information
about the subject matter or to verify facts.

You will be provided with an image or a link to an image along with a title and
accompanying text (description). Please research thoroughly and answer the following:
1. Is the image relevant to the title or description? Verify if the image and the text are about the same subject, topic, or entity. 
2. Does the image have any quality issues? Quality issues include blurred images,
Cropped images (particularly people’s heads), grainy, noisy images etc. 
3. Is the image broken or not visible?
4. Does it look like the title and description are about the same news as one of the trends already examined? 
Please provide a justification for your decision for each image.
Output format:
Image relevance: [Relevant/Irrelevant]
Image quality: [Good/Bad quality issue description]
Image integrity: [None/Issue description]
Trend duplicate: [None/provide the previous duplicate] 
Justification: [Your detailed justification]
"""
    
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information about a topic, person, or event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                        },
                        "required": ["query"],
                    },
                }
            }
        ] if use_web_search else None
    
        for item in tqdm(self.items, desc="Analyzing images"):
            if not item.image_data:
                item.analysis_result = {
                    "image_relevance": "N/A",
                    "text_relevance": "N/A", 
                    "image_quality": "N/A",
                    "image_integrity": "Image not available",
                    "justification": "Image data could not be retrieved."
                }
                continue
            
            try:
                # Converting image to base64
                image = Image.open(BytesIO(item.image_data))
                buffered = BytesIO()
                image.save(buffered, format=image.format if image.format else "JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Determine image format (JPEG is default)
                img_format = image.format.lower() if image.format else "jpeg"
                
                # Creating user message with text and image
                user_text = f"Title: {item.title}\n"
                if item.description:
                    user_text += f"Description: {item.description}\n"
                user_text += "Please analyze this image and provide your assessment. You can use web_search if you need current information about the subject."
                
                print(f"Analyzing image for '{item.title[:30]}...'")
                
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{img_format};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    # Calling GPT-4.1 API with tools if enabled
                    completion_args = {
                        "model": "gpt-4.1",
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 1000
                    }
                    
                    if use_web_search:
                        completion_args["tools"] = tools
                        completion_args["tool_choice"] = "auto"
                    
                    completion = self.openai_client.chat.completions.create(**completion_args)
                    
                    response_message = completion.choices[0].message
                    
                    if use_web_search and response_message.tool_calls:
                        for tool_call in response_message.tool_calls:
                            if tool_call.function.name == "web_search":
                                function_args = json.loads(tool_call.function.arguments)
                                search_query = function_args.get("query")
                                
                                search_results = self.web_search(search_query)
                                
                                messages.append(response_message)
                                
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": search_results
                                })
                        
                        # Get the final response after tool use
                        final_completion = self.openai_client.chat.completions.create(
                            model="gpt-4.1",
                            messages=messages,
                            temperature=0.1,
                            max_tokens=1000
                        )
                        response_text = final_completion.choices[0].message.content
                    else:
                        # No tool calls, use the direct response
                        response_text = response_message.content
                
                    # Extract analysis results
                    lines = response_text.strip().split('\n')
                    result = {}
                    
                    for line in lines:
                        if line.startswith("Image relevance:"):
                            result["image_relevance"] = line.replace("Image relevance:", "").strip()
                        elif line.startswith("Text relevance:"):
                            result["text_relevance"] = line.replace("Text relevance:", "").strip()
                        elif line.startswith("Image quality:"):
                            result["image_quality"] = line.replace("Image quality:", "").strip()
                        elif line.startswith("Image integrity:"):
                            result["image_integrity"] = line.replace("Image integrity:", "").strip()
                    
                    # Extract justification (everything after "Justification:")
                    justification_index = response_text.find("Justification:")
                    if justification_index != -1:
                        result["justification"] = response_text[justification_index + len("Justification:"):].strip()
                    else:
                        result["justification"] = "No justification provided."
                    
                    item.analysis_result = result
                    
                except Exception as e:
                    print(f"Error calling OpenAI API: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    item.analysis_result = {
                        "image_relevance": "Error",
                        "text_relevance": "Error", 
                        "image_quality": "Error",
                        "image_integrity": "Error",
                        "justification": f"Error analyzing image with OpenAI API: {str(e)}"
                    }
            
                time.sleep(1)
                
            except Exception as e:
                print(f"Error analyzing image for item '{item.title}': {e}")
                import traceback
                traceback.print_exc()
                item.analysis_result = {
                    "image_relevance": "Error",
                    "text_relevance": "Error", 
                    "image_quality": "Error",
                    "image_integrity": "Error",
                    "justification": f"Error analyzing image: {str(e)}"
                }             
        
    
    def generate_report(self):
        """Generate a report of the analysis results"""
        if not self.items:
            print("No items to report. Fetch trending module and analyze images first.")
            return pd.DataFrame()
        
        # Create DataFrame
        report_data = []
        for item in self.items:
            if item.analysis_result:
                row = {
                    "Position": item.position,
                    "Title": item.title,
                    "Description": item.description or "",
                    "Image URL": item.image_url or "",
                    "Image Relevance": item.analysis_result.get("image_relevance", "N/A"),
                    "Text Relevance": item.analysis_result.get("text_relevance", "N/A"),
                    "Image Quality": item.analysis_result.get("image_quality", "N/A"),
                    "Image Integrity": item.analysis_result.get("image_integrity", "N/A"),
                    "Justification": item.analysis_result.get("justification", "N/A"),
                }
                report_data.append(row)
        
        df = pd.DataFrame(report_data)
        return df
    
    def save_report(self, filepath="yahoo_trending_analysis.csv"):
        """Save the report to a CSV file"""
        report = self.generate_report()
        if not report.empty:
            write_header = not os.path.isfile(filepath)
            report.to_csv(filepath, mode='a', header=write_header, index=False)
            print(f"Report saved to {filepath}")
        else:
            print("No report to save.")
    
    def save_images(self, directory="yahoo_trending_images"):
        """Save the images to a directory"""
        if not self.items:
            print("No items to save. Fetch trending module first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        for item in self.items:
            if item.image_data:
                try:
                    image = Image.open(BytesIO(item.image_data))
                    filename = "".join(c if c.isalnum() or c in " .-_" else "_" for c in item.title)
                    filename = f"{item.position}_{filename[:50]}.{image.format.lower() if image.format else 'jpg'}"
                    filepath = os.path.join(directory, filename)
                    image.save(filepath)
                    print(f"Saved image to {filepath}")
                except Exception as e:
                    print(f"Error saving image for '{item.title}': {e}")
    
    def summary(self):
        """Generate a summary of the analysis results"""
        if not self.items or not all(item.analysis_result for item in self.items):
            print("No analysis results to summarize.")
            return {}
        
        total_items = len([item for item in self.items if item.analysis_result])
        relevant_images = sum(1 for item in self.items 
                            if item.analysis_result and 
                            item.analysis_result.get("image_relevance", "").lower() == "relevant")
        relevant_text = sum(1 for item in self.items 
                          if item.analysis_result and 
                          item.analysis_result.get("text_relevance", "").lower() == "relevant")
        good_quality = sum(1 for item in self.items 
                         if item.analysis_result and 
                         "good" in item.analysis_result.get("image_quality", "").lower())
        no_integrity_issues = sum(1 for item in self.items 
                               if item.analysis_result and 
                               "none" in item.analysis_result.get("image_integrity", "").lower())
        
        summary = {
            "Total Items": total_items,
            "Relevant Images (%)": round(relevant_images / total_items * 100 if total_items else 0, 2),
            "Relevant Text (%)": round(relevant_text / total_items * 100 if total_items else 0, 2),
            "Good Quality Images (%)": round(good_quality / total_items * 100 if total_items else 0, 2),
            "No Integrity Issues (%)": round(no_integrity_issues / total_items * 100 if total_items else 0, 2),
        }
        
        return summary
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None

# Function to run the analysis
def run_analysis_selenium(query="donald trump", api_key=None, use_web_search=True):
    """Run the complete analysis process using Selenium
    
    Args:
        query: Search query to use for Yahoo SERP
        api_key: OpenAI API key (optional)
        use_web_search: Whether to enable web search during analysis
    
    Returns:
        YahooTrendAnalyzerSelenium instance or None if error
    """
    try:
        analyzer = YahooTrendAnalyzerSelenium(headless=True, api_key=api_key)
        
        print("Fetching trending module...")
        success = analyzer.fetch_trending_module(query=query)
        if not success:
            print("Failed to fetch trending module.")
            analyzer.close()
            return None
        
        print(f"Found {len(analyzer.items)} trending items.")
        
        print(f"Analyzing images (web search: {'enabled' if use_web_search else 'disabled'})...")
        analyzer.analyze_images(use_web_search=use_web_search)
        
        print("Generating report...")
        report = analyzer.generate_report()
        from IPython.display import display
        display(report)
        
        print("Summary statistics:")
        summary = analyzer.summary()
        display(pd.DataFrame([summary]))
        
        print("Saving report and images...")
        analyzer.save_report()
        analyzer.save_images()
        
        analyzer.close()
        
        return analyzer
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage with web search enabled
analyzer = run_analysis_selenium(query="rod stewart", use_web_search=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




