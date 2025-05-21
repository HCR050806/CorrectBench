import time
import pprint
import requests
import json
from serpapi import GoogleSearch
#import serpapi

#AIzaSyDXbJLkDWQh24kKheDm_TMeiOQmvkpGdfA
# TODO: upload google API & web page parser
# google_search = None
# bing_search = None
# wiki_search = None


# print the JSON response from Scale SERP
#print(json.dumps(api_result.json()))
#77f381f6967d64081239cb817a7a81b5ca8631da72307916ed6ac59897aa9a34
#D3473B032C884F86A81B3E21414C523D
def google(query, cache=True, topk=1, end_year=None, verbose=False):
    assert topk >= 1
    params = {
  'api_key': '77f381f6967d64081239cb817a7a81b5ca8631da72307916ed6ac59897aa9a34',
  'q': query,
  'location':"United States",
  'page': '1',
  'num': '5'
}
    gresults = {"page": None, "title": None}

    print(gresults.get("page"))
    trial = 0
    while gresults["page"] is None and trial < 3:
        trial += 1
        if trial > 1:
            print("Search Fail, Try again...")
        #gresults = google_search(query, cache=cache, topk=topk, end_year=end_year)
        #tmp_results = requests.get('https://api.scaleserp.com/search', params)
        tmp_results = GoogleSearch(params)
        #tmp_results = serpapi.search(params)
        tmp_results = tmp_results.get_dict()   
        search_results = tmp_results.get('organic_results', [])
        print("search_results: \n",search_results)
        try:
            if search_results is None:
                continue
            # print(tmp_results['organic_results'][0]['snippet'])
            # gresults["page"] = tmp_results['organic_results'][0]['snippet']
            # gresults["page"] = tmp_results
            # gresults["title"] = query
            for i in range(min(2, len(search_results))):  # 只提取前一条
                result = search_results[i]
                gresults['title'] = result.get('title', '没有标题')
                gresults['page'] = result.get('snippet', '没有内容')
        except:
            print("trial failed")    
        time.sleep(3 * trial)

    if verbose:
        pprint.pprint(gresults)

    return gresults


def _test_google():

    queries = [
        "Who has won the American election in 2024? Trump or Haris."
    ]

    for q in queries:
        for topk in range(1, 2):
            res = google(q, verbose=True, cache=False, topk=topk)
            # print(res['organic_results'][0]['snippet'])
            # print(res['organic_results'][0]['snippet'])
            print(f"[{res.get('title', '')}] {res.get('page', '')}")


if __name__ == "__main__":
    _test_google()