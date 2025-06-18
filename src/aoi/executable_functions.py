import requests

def do_search_wiki(query, num_results=10):
    # Use Wikipedia's free API
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": num_results
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    results = []
    if "query" in data and "search" in data["query"]:
        for item in data["query"]["search"]:
            results.append({
                "title": item["title"],
                "snippet": item["snippet"],
                "url": f"https://en.wikipedia.org/wiki/{item['title']}"
            })
    
    returned_values = "\n".join([f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}" for result in results])
    return returned_values

def do_search_news(query, num_results=10):
    # Use NewsAPI's free endpoint
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": num_results,
        "apiKey": "90f52aa7b0b04cbea5297357ca2c9f17"  # Users need to get a free API key from newsapi.org
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    results = []
    if "articles" in data:
        for article in data["articles"][:num_results]:
            results.append({
                "title": article["title"],
                "description": article["description"],
                "url": article["url"]
            })
    returned_values = "\n".join([f"Title: {result['title']}\nURL: {result['url']}\nDescription: {result['description']}" for result in results])
    return returned_values

def do_search_cookbook(query, num_results=5, locale="EN"):
    # Use JD API for recipe search
    url = "https://way.jd.com/jisuapi/search"
    params = {
        "keyword": query,
        "num": min(num_results, 20),  # API limit is 20
        "appkey": "da39dce4f8aa52155677ed8cd23a647"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    results = []
    if "result" in data and "result" in data["result"] and "list" in data["result"]["result"]:
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
    
    returned_values = "\n\n---\n\n".join([f"Title: {result['title']}\nURL: {result['url']}\nDescription: {result['description']}" for result in results])
    return returned_values

def test_search_wiki():
    # Test basic wiki search
    results = do_search_wiki("Python programming language")
    assert len(results) > 0
    assert "title" in results[0]
    assert "snippet" in results[0]
    
    # Test with custom num_results
    results = do_search_wiki("Python programming language", num_results=3)
    assert len(results) <= 3
    print("--------------------------------")
    print("search Wiki Results:")
    print(results)
    
    # Test with empty query
    results = do_search_wiki("")
    assert len(results) == 0
    
    print("All wiki search tests passed!")

def test_search_news():
    # Test basic news search
    results = do_search_news("artificial intelligence")
    assert len(results) > 0
    assert "title" in results[0]
    assert "description" in results[0]
    assert "url" in results[0]
    
    # Test with custom num_results
    results = do_search_news("artificial intelligence", num_results=5)
    assert len(results) <= 5
    print("--------------------------------")
    print("search News Results:")
    print(results)
    
    # Test with empty query
    results = do_search_news("")
    assert len(results) == 0
    
    print("All news search tests passed!")

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
    test_search_wiki()
    test_search_news()
    test_search_cookbook()