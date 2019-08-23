import facebook

graph = facebook.GraphAPI(access_token="EAAhbslbBjxIBAKMnqHy4KMr2MJ6FxAOAwZA70mztFPH4731OY3NJpo9BNRDEhm2Ucin6hBQ5bvd9TZCE3ZBAsjzqsFfICUwZBbEjuoFoJiEUEqAhsz9I8QexqaOvxzWoVFzf7MkNZCqmhq6QWYdNcVG21chAdIFdXZB5z0xnBZCNFwrTvxnvocevnF7dOAHgVP1DSFdejjWFEBERpgaVkX8", version="3.3")


group_posts = graph.get_object(id="916799215197938/feed")

# for item in group_posts:
#     print(item)


if "data" in group_posts:
    data = group_posts["data"]
    for item in data:
        # print(item['id'])

        post_id = item.get('id')
        post_message = item.get('message')
        if not post_id:
            continue

        print(f"Post message: {post_message}")

        post_comments = graph.get_object(id=post_id, fields="comments")
        for comment in post_comments:
            if "comment" in comment:
                comments_data = post_comments['comments']['data']
                for comment_item in comments_data:
                    comment_message = comment_item.get('message')
                    comment_id = comment_item.get('id')
                    comment_created_time = comment_item.get('created_time')

                    print(f"\tComment: {comment_message}")
