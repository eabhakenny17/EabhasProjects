# LLM as a Jury

A command-line simulation that asks independent LLM jurors to assess the content in a file and tallies their votes.

## Requirements

- Python 3.10 or newer
- An API endpoint compatible with the OpenAI Chat Completions request/response format
- An API key and model name supported by that endpoint

## Install

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Put the provider configuration in a `.env` file in this project directory. The CLI loads that file automatically. Do not commit API keys:

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
LLM_MODEL=your-model-name
```

## Run instructions

Provide a UTF-8 `.txt` case file as the positional argument eg. `examples/sample_case.txt`.

```powershell
mock-jury .\examples\sample_case.txt --jurors 6
```

OR

```powershell
python -m llm_as_a_jury .\examples\sample_case.txt --jurors 6
```

OR (for jury2.py)

python -m llm_as_a_jury.jury2 .\src\llm_as_a_jury\jury.py --reviewers 1 --timeout 600

-> reviewers instead of jurors as the LLM instances are referred to as jurors in jury.py and reviewers in jury2.py.
-> manually overriding timeout (set to 300 by default), especially important when using less powerful models.

## Tests

```powershell
python -m unittest discover -s tests -v
```


# Step by Step

1) The command with the required text file is written in the terminal.

2) cli.py accepts the path written in the terminal and loads configuration for .env. argparse defines it as case_file.

3) args.case_file is passed to load_case() in case_file.py. load_case returns the file contents as a string as case_text.

4) case_text is passed to run_jury() in jury.py. The LLM is called once per simulated juror. Each request contains the same case and instructions, as well as an identifier. The instruction asks for a JSON object with a verdict and rationale. 

5) The LLM is told to only use case facts.

6) jury.py validates each response. To be a valid response, the response must be valid JSON, contain an accepted vote value and include a rationale.

6) jury.py tallies the votes. The CLI prints the tally and rationales.



7) jury2.py, a file (text or code) review framework, works more independently. It asks for a subjective 1 - 5 score and rationale based on the file's text. It does not execute or test the code it reviews. All that is required is to reference jury2.py and the file to be reviewed, assuming that project provider settings have been configured in .env and dependencies are installed.


# Note
This is a derivative lite rendition of a project I completed individually while working at Liberty IT. Therefore, this does not contain the full extent of the potential functionality. Additionally, there is a limitation on agent ability due to the fact that I am using a free model (local instance of Ollama), however you can swap in a paid model in your own usage of this project.