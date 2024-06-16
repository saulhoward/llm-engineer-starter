# Submission report

```bash
python submission.py --path=data/inpatient_record.pdf
```

[Example output](https://github.com/saulhoward/llm-engineer-starter/blob/main/data/example-output.json)

## Decisions

My initial thoughts and assumptions were:

- The first problem is the need to split the document into relevant sections.
- The sections in this report represent medical encounter events, each occurring at a specific time.
- When faced with this problem before, I opted to use semantic distance to split the document into sections (using embeddings). I don't think this approach would work here, because each medical encounter is highly similar semantically.

I therefore decided to try this approach:

- Page through the document text, asking the LLM to identify "event boundaries".
- Given the line number of an "event boundary", representing the beginning of a new medical encounter, split the document at those line numbers.

Once I had a list of separate "medical encounters", I would ask the LLM to identify structured information within the text and compile that into a pre-defined schema. Reviewing the document, likely candidates for structured information were:

- Timestamp of the encounter
- List of "medical findings" by the physician
- List of prescription medicines mentioned

Other structured information will certainly be in the reports. Of course, the exact information that should be pulled out will depend upon the requirements of the downstream systems making use of the schema.

## Development process

The first gotcha was that the document, at 60 pages, was too long for a single call to Document AI. I solved this by:

- Streaming the PDF into memory.
- Creating in-memory PDFs for every 10 pages.
- Sending the 10 page PDF to DocumentAI to extract text.
- Concatenating the text into a single list of lines.

To identify the "medical encounters" within the complete text:

- Send a chunk of lines from the text to the LLM
- Ask the LLM to identify which lines represent a boundary (the beginning of a new medical encounter)

With the medical encounters now separated:

- Prompt the LLM to identify a timestamp for the encounter
- Prompt the LLM to list medical findings in the encounter
- Prompt the LLM to list prescriptions mentioned in the encounter

With the information now structured:

- Map the encounter information into fields in a Pydantic model.
- Return the model as JSON.

## LLM choice

- I used `gpt-4-turbo` for development
- I included an abstraction layer over OpenAI and Ollama. I've found this useful as a sanity check.
- Results from `llama3:8b` were not good, as expected.
- Results from `llama3:70b` do work, but on my hardware the model runs too slowly.
- I opted not to use OpenAI specific features, like tools/functions. I've had good experience using these to generate JSON, but that wasn't necessary here.

## Results

- The splitting approach seems to work. Further research/experimentation may point to a better solution.
- The "medical findings" are not differentiated enough. The prompt causes confusion between, e.g. the patient's history and their family history. More detailed prompting is needed.

## Further thoughts

- To improve the encounter boundary finding, overlapping sections of text should be sent to the LLM.
- Repeated information like patient DOB will probably cause confusion. This common information could be detected in an initial passthrough and used to catch errors.
- Deduplication of repeated medical encounters or other text chunks in the report.
- OpenAI tools/functions could be used to chain together calls, e.g. to first detect the type of medical encounter and then look for findings, prescriptions etc.
- The abstraction layer could be extended to test with Google models etc.
