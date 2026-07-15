Open source claude skill idea 

A skill that helps claude read 1 to 1million pdfs 

Document ingestion
- pdf
- doc
- md/text files 
Document Parsing
- reading order
- extract outline (mymupdf4llm, upgrade to docling/mineru) 
- table cells/image cells
- paragraph chunker 
- NER
	- Might need a dedupe pass (does open source exist for this?) 
		- look into dedupe libraries 
Document routing 
- send tables / images to small llm for descriptions
Embedding 
-embedding selector (abstraction) 
 - default: potion 32 embeddings (cheap VERY fast, not the best retrieval) 
	- gemma 4 embeddings (good balance between cheap and retrieval quality) 
	- Qwen3 / Nemotron (upgrade/expensive/takes time/needs api key) 
	- use turbovec to store embeddings cheaply for retrieval (see sister flashvec library for inspo) 
Indexing 
- bm25
- Clustering Flash kmeans (mlx) 
	- c-tf-idf → get vocab → lightweight LLM → get topical labels 
	- create topical index over chunks
- with NER → person, place, year index over chunks 
Retrieval 
- optional reranker?
Save in embedded database? vector sql? Lancedb? chromadb? duckdb plugin? sql mcp? Helix? 

Expose claude with search tools 
- terms, topics, persons, places 
Create set of baked in queries 
- Who are the most common names 
- 

concerns: speed, ease of setup,




# relevant local projects
- deepreader (could have some useful reading flows, but more for single docs) 
- flashvec/flashvec-speedup (embedding storage ideas)
- harnessx (experiment tracking, could be relevant for journalist agent observation tracking)
- response comparision (cluster and topic labeling with potion and mlx)
- document-parsing (has different document parsing comparisions, dont extract the model built there though) 
- paper-search-cmp 


# possibliy relevant external sources
https://github.com/jamditis/claude-skills-journalism
https://huggingface.co/spaces/fdaudens/ai-journalism-skills
Though these are super light weight mostly md files of lists or sources, not actual code

asta (scholarly search mcp)
