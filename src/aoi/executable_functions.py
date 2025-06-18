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

if __name__ == "__main__":
    test_search_wiki()
    test_search_news()