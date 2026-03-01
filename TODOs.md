[DONE TASKS]
- Migrate the /upload-data background task from case_v1 router to a celery task --> (DONE)
- See the notes and browsing extracting from Ufdringester.py and merge it with the old UfdrParser.py --> (DONE)

Validate all the classifications by Ollama (Sentiments, Toxicity etc)
Add an option for the user to use NER model for entity extraction or Llama
Add another section named "semantic labels" like [travel, sports, ...] and the user can rag query from those label(s)
The user can search with photo (detector) on per case or all cases level
We need to have detector profile (multiple faces and objects) for each case
Add detector as a filter in the media files explorer
Add NSFW filter in the media explorer
Add embeddings for descriptions of photos/video (optional) and for audio it is necessary and user will be able to search by that in the media files explorer
Add the labels filter in the media files explorer to search from the multi media description classes
Add shouting or not shouting filter audio and video transcription
In extraction of every 30th frame we need to look into ffmpeg to only pick unique frame and no duplicate frame
Add the same bottom section like location, top entities etc in the media files explorer
- Resolve the following neo4j error:    
    2025-11-24 00:13:16,272 - INFO - Connecting to Neo4j at bolt://localhost:7687
    2025-11-24 00:13:16,277 - ERROR - Failed to verify Neo4j connection: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
    Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 61] Connection refused)
    Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 61] Connection refused)
    2025-11-24 00:13:16,277 - ERROR - Failed to connect to Neo4j: Cannot connect to Neo4j: Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
    Failed to establish connection to ResolvedIPv6Address(('::1', 7687, 0, 0)) (reason [Errno 61] Connection refused)
    Failed to establish connection to ResolvedIPv4Address(('127.0.0.1', 7687)) (reason [Errno 61] Connection refused)
- Integrate the "intfloat/multilingual-e5-large" for embeddings generations