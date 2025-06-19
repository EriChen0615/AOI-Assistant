import requests
import os

def do_search_bocha(query, num_results=10, freshness="noLimit", include=None, answer=False, stream=False):
    """
    Search using Bocha AI search API
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return (max 50)
        freshness (str): Time range - oneDay, oneWeek, oneMonth, oneYear, noLimit
        include (str): Site restrictions (e.g., "qq.com|m.163.com")
        answer (bool): Whether to use AI for answering
        stream (bool): Whether to use streaming response
    """
    # Read API key from config file
    api_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "bocha_api_key")
    try:
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        return "Error: Bocha API key not found. Please check configs/bocha_api_key file."
    
    url = "https://api.bochaai.com/v1/ai-search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "freshness": freshness,
        "count": min(num_results, 50),  # API limit is 50
        "answer": answer,
        "stream": stream
    }
    
    if include:
        payload["include"] = include
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Parse the messages array from the response
        if "messages" in data:
            for message in data["messages"]:
                if message.get("content_type") == "webpage":
                    # Parse webpage content
                    try:
                        import json
                        webpage_data = json.loads(message["content"])
                        if "value" in webpage_data:
                            for item in webpage_data["value"]:
                                results.append({
                                    "title": item.get("name", ""),
                                    "snippet": item.get("snippet", ""),
                                    "url": item.get("url", ""),
                                    "source": item.get("siteName", ""),
                                    "summary": item.get("summary", "")
                                })
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing webpage content: {e}")
                        continue
                
                # elif message.get("content_type") == "baike_pro":
                #     # Parse baike content
                #     try:
                #         import json
                #         baike_data = json.loads(message["content"])
                #         for item in baike_data:
                #             if "modelCard" in item and "module_list" in item["modelCard"]:
                #                 for module in item["modelCard"]["module_list"]:
                #                     for sub_item in module.get("item_list", []):
                #                         if "data" in sub_item and "card" in sub_item["data"]:
                #                             card = sub_item["data"]["card"]
                #                             results.append({
                #                                 "title": card.get("title", ""),
                #                                 "snippet": card.get("dynAbstract", ""),
                #                                 "url": item.get("url", ""),
                #                                 "source": "百科",
                #                                 "summary": card.get("abstract_info", "")
                #                             })
                #     except (json.JSONDecodeError, KeyError) as e:
                #         print(f"Error parsing baike content: {e}")
                #         continue
                
                # elif message.get("content_type") == "medical_common":
                #     # Parse medical content
                #     try:
                #         import json
                #         medical_data = json.loads(message["content"])
                #         for item in medical_data:
                #             if "modelCard" in item and "subitem" in item["modelCard"]:
                #                 for sub_item in item["modelCard"]["subitem"]:
                #                     results.append({
                #                         "title": sub_item.get("title", "").replace("<em>", "").replace("</em>", ""),
                #                         "snippet": sub_item.get("content", ""),
                #                         "url": sub_item.get("wapUrl4Resin", ""),
                #                         "source": f"{sub_item.get('hospital', '')} - {sub_item.get('doctorName', '')}",
                #                         "summary": sub_item.get("content", "")[:200] + "..." if len(sub_item.get("content", "")) > 200 else sub_item.get("content", "")
                #                     })
                #     except (json.JSONDecodeError, KeyError) as e:
                #         print(f"Error parsing medical content: {e}")
                #         continue
        
        # Limit results to requested number
        results = results[:num_results]
        
        # Format results similar to the original functions
        if results:
            returned_values = "\n\n---\n\n".join([
                f"Title: {result['title']}\nURL: {result['url']}\nSource: {result['source']}\nSnippet: {result['snippet']}\nSummary: {result['summary']}" 
                for result in results
            ])
        else:
            returned_values = "No results found."
        
        return returned_values
        
    except requests.exceptions.RequestException as e:
        return f"Error making request to Bocha API: {str(e)}"
    except KeyError as e:
        return f"Error parsing Bocha API response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def do_search_bocha_websearch(query, num_results=10, freshness="noLimit", include=None, answer=False, stream=False):
    pass
# Replace the old search functions with Bocha search
def do_search_wiki(query, num_results=10):
    """
    Search Wikipedia using Bocha AI search
    """
    return do_search_bocha(f"site:wikipedia.org {query}", num_results)

def do_search_news(query, num_results=10):
    """
    Search news using Bocha AI search
    """
    return do_search_bocha(f"news {query}", num_results, freshness="oneWeek")

def do_search_cookbook(query, num_results=5, locale="EN"):
    # Use JD API for recipe search
    url = "https://way.jd.com/jisuapi/search"
    params = {
        "keyword": query,
        "num": min(num_results, 20),  # API limit is 20
        "appkey": "da39dce4f8aa52155677ed8cd23a6470"
    }
    
    print(f"DEBUG: Making API call to {url}")
    print(f"DEBUG: Parameters: {params}")
    
    response = requests.get(url, params=params)
    print(f"DEBUG: Response status code: {response.status_code}")
    
    data = response.json()
    print(f"DEBUG: API response: {data}")
    
    results = []
    if "result" in data and "result" in data["result"] and "list" in data["result"]["result"]:
        print(f"DEBUG: Found recipe list with {len(data['result']['result']['list'])} items")
        for item in data["result"]["result"]["list"][:num_results]:
            # Extract ingredients
            ingredients = []
            if "material" in item:
                for material in item["material"]:
                    ingredients.append(f"{material.get('mname', '')} {material.get('amount', '')}")
            
            # Extract cooking steps
            steps = []
            if "process" in item:
                for i, process in enumerate(item["process"], 1):
                    steps.append(f"{i}. {process.get('pcontent', '')}")
            
            # Format the result based on locale
            if locale == "ZH":
                recipe_info = f"菜谱: {item.get('name', '')}\n"
                recipe_info += f"准备时间: {item.get('preparetime', '')}\n"
                recipe_info += f"烹饪时间: {item.get('cookingtime', '')}\n"
                recipe_info += f"份量: {item.get('peoplenum', '')}\n"
                recipe_info += f"标签: {item.get('tag', '')}\n\n"
                
                if ingredients:
                    recipe_info += "食材:\n" + "\n".join(ingredients) + "\n\n"
                
                if steps:
                    recipe_info += "烹饪步骤:\n" + "\n".join(steps) + "\n\n"
                
                recipe_info += f"描述: {item.get('content', '')}"
            else:
                recipe_info = f"Recipe: {item.get('name', '')}\n"
                recipe_info += f"Preparation Time: {item.get('preparetime', '')}\n"
                recipe_info += f"Cooking Time: {item.get('cookingtime', '')}\n"
                recipe_info += f"Serves: {item.get('peoplenum', '')}\n"
                recipe_info += f"Tags: {item.get('tag', '')}\n\n"
                
                if ingredients:
                    recipe_info += "Ingredients:\n" + "\n".join(ingredients) + "\n\n"
                
                if steps:
                    recipe_info += "Cooking Steps:\n" + "\n".join(steps) + "\n\n"
                
                recipe_info += f"Description: {item.get('content', '')}"
            
            results.append({
                "title": item.get("name", ""),
                "description": recipe_info,
                "url": item.get("pic", "")
            })
    else:
        print(f"DEBUG: No recipe list found in response")
        print(f"DEBUG: Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        if "result" in data:
            print(f"DEBUG: Result keys: {list(data['result'].keys()) if isinstance(data['result'], dict) else 'Not a dict'}")
    
    print(f"DEBUG: Processed {len(results)} results")
    
    returned_values = "\n\n---\n\n".join([f"Title: {result['title']}\nURL: {result['url']}\nDescription: {result['description']}" for result in results])
    print(f"DEBUG: Final returned values length: {len(returned_values)}")
    return returned_values

def test_search_bocha():
    # Test basic Bocha search with Chinese query
    results = do_search_bocha("西瓜的功效与作用")
    print("--------------------------------")
    print("Bocha Search Results (Chinese):")
    print(results)
    
    # Test with custom num_results
    results = do_search_bocha("Python programming language", num_results=5)
    print("--------------------------------")
    print("Bocha Search Results (English, limited):")
    print(results)
    
    # Test with freshness filter
    results = do_search_bocha("latest technology news", freshness="oneWeek")
    print("--------------------------------")
    print("Bocha Search Results (recent):")
    print(results)
    
    # Test with empty query
    results = do_search_bocha("")
    print("--------------------------------")
    print("Bocha Search Results (empty query):")
    print(results)
    
    print("All Bocha search tests completed!")

def test_search_wiki():
    # Test Wikipedia search using Bocha
    results = do_search_wiki("Python programming language")
    print("--------------------------------")
    print("Wikipedia Search Results (via Bocha):")
    print(results)
    
    # Test with custom num_results
    results = do_search_wiki("Python programming language", num_results=3)
    print("--------------------------------")
    print("Wikipedia Search Results (limited):")
    print(results)
    
    # Test with empty query
    results = do_search_wiki("")
    print("--------------------------------")
    print("Wikipedia Search Results (empty query):")
    print(results)
    
    print("All wiki search tests completed!")

def test_search_news():
    # Test news search using Bocha
    results = do_search_news("artificial intelligence")
    print("--------------------------------")
    print("News Search Results (via Bocha):")
    print(results)
    
    # Test with custom num_results
    results = do_search_news("artificial intelligence", num_results=5)
    print("--------------------------------")
    print("News Search Results (limited):")
    print(results)
    
    # Test with empty query
    results = do_search_news("")
    print("--------------------------------")
    print("News Search Results (empty query):")
    print(results)
    
    print("All news search tests completed!")

def test_search_cookbook():
    # Test basic cookbook search
    results = do_search_cookbook("蛋炒饭")
    assert len(results) > 0
    print("--------------------------------")
    print("search Cookbook Results:")
    print(results)
    
    # Test with custom num_results
    results = do_search_cookbook("红烧肉", num_results=5)
    assert len(results) <= 5
    print("--------------------------------")
    print("search Cookbook Results (limited):")
    print(results)
    
    # Test with empty query
    results = do_search_cookbook("")
    assert len(results) == 0
    
    print("All cookbook search tests passed!")

if __name__ == "__main__":
    test_search_bocha()
    test_search_wiki()
    test_search_news()
    test_search_cookbook()