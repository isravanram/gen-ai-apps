
def convert_text_to_sql(query_text):
    try:
        start_time = time.time()
        # Load local model path
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        # Static subpath (relative to current directory)
        subpath = "local_models/arctic-text2sql/models--Snowflake--Arctic-Text2SQL-R1-7B/snapshots"

        # Combine to get absolute path
        base_path = os.path.join(current_dir, subpath)
        print(f"Base path for model: {base_path}")
        # base_path = "/Users/shrav/Documents/Taippa/generate-v2/generate/app/pipelines/local_models/arctic-text2sql/models--Snowflake--Arctic-Text2SQL-R1-7B/snapshots"
        snapshot_hash = os.listdir(base_path)[0]
        model_path = os.path.join(base_path, snapshot_hash)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Input: natural language question and schema
        # natural_language_question = "Give me best customers by revenue in the last 30 days"
        schema_context = """
        Database schema:
        Table: customers (id, name, revenue, signup_date)
        """

        natural_language_question = query_text

        schema_context = """"

        Database schema with column descriptions:
        Table: influencers (
        id                          character varying(100000) -- Unique identifier for the influencer
        instagram_url               character varying(100000) -- URL of the influencer's Instagram profile
        instagram_followers_count   character varying(100000) -- Number of followers on Instagram
        instagram_username          character varying(100000) -- Instagram username/handle
        instagram_bio               character varying(100000) -- Bio text from the Instagram profile
        influencer_type             character varying(100000) -- Type/category of influencer (e.g., fashion, tech)
        influencer_location         character varying(100000) -- Location of the influencer
        instagram_post_urls         character varying(100000) -- List of URLs to the influencer's Instagram posts
        business_category_name      character varying(100000) -- Main business category of the influencer (Tag provided by Instagram for Business profile Eg: "Personal blog","Digital creator","Reel creator" etc)
        full_name                   character varying(100000) -- Full name of the influencer
        instagram_follows_count     character varying(100000) -- Number of accounts the influencer follows
        created_time                character varying(100000) -- Timestamp when the influencer was added to the database
        instagram_hashtags          character varying(100000) -- Hashtags used by the influencer for different posts (stored as list of strings)
        instagram_captions          character varying(100000) -- Captions from the influencer's posts (stored as list of strings)
        instagram_video_play_counts character varying(100000) -- Number of plays for video posts (stored as list of strings)
        instagram_likes_counts      character varying(100000) -- Number of likes on posts (stored as list of strings)
        instagram_comments_counts   character varying(100000) -- Number of comments on posts (stored as list of strings)
        instagram_video_urls        character varying(100000) -- URLs of video posts
        instagram_posts_count       character varying(100000) -- Total number of posts by the influencer
        external_urls               character varying(100000) -- List of external links provided by the influencer
        instagram_profile_pic       character varying(100000) -- URL of the influencer's profile picture
        influencer_nationality      character varying(100000) -- Nationality of the influencer (Include country names)
        targeted_audience           character varying(100000) -- Target audience group for the influencer (fixed category values among this: ["gen-z","gen-y", "gen-x"])
        targeted_domain             character varying(100000) -- Domain or industry targeted by the influencer (fixed categoru values among this: ["food", "fashion", "fitness", "gaming", "education", "automotive", "finance", "art"])
        profile_type                character varying(100000) -- Type of profile (fixed category values among this: ["person","group"])
        email_id                    character varying(100000) -- Email address of the influencer (if not available value is "NA")
        twitter_url                 character varying(100000) -- URL of the influencer's Twitter profile
        snapchat_url                character varying(100000) -- URL of the influencer's Snapchat profile
        phone                       character varying(100000) -- Phone number of the influencer ( if not available value is "NA")
        linkedin_url                character varying(100000) -- URL of the influencer's LinkedIn profile
        tiktok_url                  character varying(100000) -- URL of the influencer's TikTok profile
        )
        """

        # Format prompt
        prompt = f"{schema_context}\n\nQuestion: {natural_language_question}\nSQL:"

        # Tokenize with padding and attention mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        # Generate output with attention mask
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                # temperature=0.3,    # Remove or set do_sample=True if you want to use temperature
                do_sample=False,     # Set to True if you want to use temperature
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the result
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract SQL query after "SQL:"
        sql_start = generated_text.find("SQL:") + len("SQL:")
        sql_query = generated_text[sql_start:].strip()


        end_time = time.time()
        # Print the results
        time_taken = end_time - start_time
        print(f"Time taken to generate SQL query: {time_taken:.2f} seconds")

        print("\nGenerated SQL Query:")
        print(sql_query)
    
    except Exception as e:
        print(f"Error during SQL generation: {e}")
        sql_query = None
    
    return sql_query

