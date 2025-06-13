import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv()


username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")


main_url="https://discourse.onlinedegree.iitm.ac.in"  
fro=datetime(2025,1,1)
to=datetime(2025,4,14)
session_file=t='cookie.json'
CATEGORY_ID = 34


CATEGORY_JSON_URL= f"{main_url}/c/courses/tds-kb/{CATEGORY_ID}.json"

def session_login(p):
    browser=p.chromium.launch()
    context=browser.new_context()
    page=context.new_page()
    page.goto(f"{main_url}/login")
    page.wait_for_selector('input#login-account-name').fill("23f2001565@ds.study.iitm.ac.in")
    page.wait_for_selector('input#login-account-password').fill("bhanu@23f2001565")
    page.wait_for_selector("button#login-button").click()
    page.wait_for_url(main_url)
    context.storage_state(path=session_file)
    browser.close()
    
def authorized(page):
    try:
        page.goto(CATEGORY_JSON_URL,timeout=10000)
        page.wait_for_selector('pre',timeout=5000)
        data=json.loads(page.inner_text("pre"))
        # with open("data/discourse.json",'w') as f:
        #     json.dump(data,f,indent=2)
        return True
    except (TimeoutError, json.JSONDecodeError):
        return False
    
def scrap(p):
    b=p.chromium.launch()
    c=b.new_context(storage_state=session_file)
    page=c.new_page()
    all_topics=[]
    p=0
    while True:
        p_url=f"{CATEGORY_JSON_URL}?page={p}"
        page.goto(p_url)
        try:
            data=json.loads(page.inner_text("pre"))
        except:
            data=json.loads(page.content())
        topics= data.get("topic_list",{}).get("topics",[])
        if not topics:
            break
        all_topics.extend(topics)
        p+=1
        print(f"page{p} is scand succussefuly ✅")
    print(len(all_topics))
    filtered_post=[]
    d={True:0,False:0}
    for t in all_topics:
        created_at=datetime.strptime(t["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        d[fro <= created_at <= to]+=1
        if fro <= created_at <= to :
            t_url= f"{main_url}/t/{t['slug']}/{t['id']}.json"
            page.goto(t_url)
            try:
                data1=json.loads(page.inner_text("pre"))
            except:
                data1=json.loads(page.content())
            posts=data1.get("post_stream",{}).get("posts",[])
            answer=data1.get("accepted_answer", data1.get("accepted_answer_post_id"))
                   
            reply_counter={}
            for post in posts:
                reply_to = post.get("reply_to_post_number")
                if reply_to is not None:
                    reply_counter[reply_to] = reply_counter.get(reply_to, 0) + 1
            
            for post in posts:
                filtered_post.append({
                    "topic_id":t['id'],
                    "topic_title":t.get('title'),
                    "category_id": t.get("category_id"),
                    "tags": t.get("tags", []),
                    "post_id": post["id"],
                    "post_number": post["post_number"],
                    "author": post["username"],
                    "created_at": post["created_at"],
                    "updated_at": post.get("updated_at"),
                    "reply_to_post_number": post.get("reply_to_post_number"),
                    "is_reply": post.get("reply_to_post_number") is not None,
                    "reply_count": reply_counter.get(post["post_number"], 0),
                    "like_count": post.get("like_count", 0),
                    "is_accepted_answer": post["id"] == answer,
                    "mentioned_users": [u["username"] for u in post.get("mentioned_users", [])],
                    "url": f"{main_url}/t/{t['slug']}/{t['id']}/{post['post_number']}",
                    "content": BeautifulSoup(post["cooked"], "html.parser").get_text()
                })
    print(d)
    os.makedirs("data", exist_ok=True)
    with open("data/raw_data.json",'w') as f:
        json.dump(filtered_post,f,indent=4)
    b.close()
    
    
        
    

with sync_playwright() as p:
    if not os.path.exists(session_file):
        session_login(p)
    else:
        print(f"{session_file} is there")
        b=p.chromium.launch()
        c=b.new_context(storage_state=session_file)
        page=c.new_page()
        if  authorized(page):
            print("succuse")
        else:
            print("No")
            session_login(p)    
        b.close()
    scrap(p)
        