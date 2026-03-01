- Optimizations in the ingest_ufdr_file function:

    - Optimizations in the process_main_xml function:

        - In the process_media_files function of the UfdrIngester class on line no. 462 of @ufdr_ingester.py we are still inserting the media documents in mongo db one by one instead of batches. Need to optimize this setup by doing bulk inserting of media documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes) 
        
        - Similarly in the scan_directory_for_media_files function of the UfdrIngester class on line no. 711 of @ufdr_ingester.py we are still inserting the media documents in mongo db one by one instead of batches. Need to optimize this setup by doing bulk inserting of media documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes)
        
        - In the extract_embedded_content_from_xml function of the UfdrIngester class it is calling functions like extract_embedded_from_element and extract_base64_from_field which are in turn calling the process_base64_content function which is again inserting the media record one by one line no. 928 of @ufdr_ingester.py. Need to optimize this setup by doing bulk inserting of embedded documents in batches in mongodb instead single insertion one by one (Need to use mongodb bulk writes) 
        
        - Then after the above three functions in the same process_main_xml function we are traversing all modelType(s)/model(s) (like messages, calls, contacts, emails etc) and inserting them all one by one in mongodb. Need to optimize this setup by doing bulk inserting of media documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes)

    - In the process_messages function both for .xml and .json files for all its messages we are doing insertion one by one on lines no. 2250 and 2337. Need to optimize this setup by doing bulk inserting of messages documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes)

    - In the process_chats function for .txt, .xml and .json files for all its chat records messages we are doing insertion one by one on lines no. 2483, 2579 and 2668. Need to optimize this setup by doing bulk inserting of chat messages documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes)

    - In the process_contacts function both for .xml and .json files for all its contact records we are doing insertion one by one on lines no. 2722 and 2748. Need to optimize this setup by doing bulk inserting of contact documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes)

    - In the process_emails function both for .eml files for all its email records we are doing insertion one by one on line no. 2841. Need to optimize this setup by doing bulk inserting of email documents in batches in mongodb instead single insertion one by one. (Need to use mongodb bulk writes)

(Need to parallelize the above functions of the UfdrIngester module if possible)


- Optimizations in the analyzer.process_documents() function.

    - In the process_documents_batched() function we are instantiating the AsyncLlamaClient class's instance everytime this method is called. We need to add this AsyncLlamaClient with all of its necessary parameters in the model_registry module and load it from there as we are doing with the classifier, toxic and emotion clients. 