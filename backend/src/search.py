import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")


def google_search(query, api_key, cse_id, num_results=10):
    """
    Performs a Google Custom Search and returns a list of search results.

    Args:
        query (str): The search query string.
        api_key (str): Your Google API key.
        cse_id (str): Your Custom Search Engine ID.
        num_results (int): The number of search results to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a search result.
               Each dictionary will contain keys like 'title', 'link', and 'snippet'.
        or
        None: If there is an error.
    """
    try:
        # Authenticate with Google API
        service = build("customsearch", "v1", developerKey=api_key)

        # Execute the search query
        res = (
            service.cse()
            .list(q=query, cx=cse_id, num=num_results)
            .execute()
        )

        # Extract relevant information from the search results
        results = []
        if "items" in res:
            for item in res["items"]:
                results.append(
                    {
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item["snippet"],
                    }
                )
        return results

    except HttpError as e:
        print(f"An error occurred: {e}")
        return None
    except Exception as e:
      print(f"An unexpected error occurred: {e}")
      return None


def search_engine(search_query, top_k=4):
    """Google search tool"""
    search_results = google_search(search_query, google_api_key, google_cse_id, num_results=top_k)

    if search_results:
        output = ""
        for result in search_results:
            output += f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n"
        return output
    else:
        return "No results found."
