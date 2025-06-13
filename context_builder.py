# # context_builder.py
# def build_context(results):
#     context = ""
#     sources = []
#     for r in results:
#         snippet = r["combined_text"][:700]
#         context += f"- {r['topic_title']} (score: {r['score']:.3f}):\n{snippet}\n\n"
#         sources.append({
#             "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{r['topic_id']}/{r['root_post_number']}",
#             "text": r["topic_title"]
#         })
#     return context, sources

# from collections import OrderedDict

def build_context(results):
    context = ""
    sources = []
    seen_urls = set()

    for res in results:
        title = res["topic_title"]
        topic_id = res["topic_id"]
        post_numbers = res["post_numbers"]
        combined_text = res["combined_text"]

        context += f"- {title}:\n{combined_text[:500]}\n\n"

        for post_number in post_numbers:
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/{post_number}"
            if url not in seen_urls:
                sources.append({"url": url, "text": title})
                seen_urls.add(url)

    return context.strip(), sources

