# SQA

## Overview
This is an end-to-end service that:
1. Generates trends/queries
2. Fetches SRP data for each trend
3. Evaluates the data (blocklist, image check, etc.)
4. Generates a combined report per query
5. Sends the report via email

## Project Structure

```
SQA/
    main.py                      # Entry point for the workflow
    requirements.txt             # Python dependencies
    env_settings.yaml            # External API endpoints or URLS
    README.md                    # Project documentation
    logs/                        # Log files
    resource/                    # Blocklists, prompts, etc.
        blocklist/               
            drugs.txt
            fraud.txt
            hate.txt
            other.txt
            piracy.txt
            self-harm.txt
            sexual.txt
            violence.txt
            weapons.txt
        prompt/
            system/
                example.txt      
            user/
                example.txt      
    service/
        filter_resource.py       # Filtering logic for SRP resources
        llm.py                   # LLM calls
        logger_setup.py          # Logging setup
        output_formatter.py      # Formats and prettifies evaluation results
        read_txt_file.py         # Utility to read text files
        send_email.py            # Email sending function
        evaluations/             # Rule based logic
            __init__.py
            blocklist.py         
        fetchers/
            __init__.py
            srp_fetcher.py       # Fetches SRP in png/html/text
            trend_fetcher.py     # Generates trends
    static/
        html/                    # Cached HTML files from SRP
        img/                     # Cached PNG screenshots from SRP
```

## Setup Instructions

1. **Clone the repository**

2. **Set up Python environment**

   [option 1: with pyenv]
   - Install [pyenv](https://github.com/pyenv/pyenv) if not already installed
   - Create a new Python environment (e.g. Python 3.10):
     ```sh
     pyenv install 3.10.13
     pyenv virtualenv 3.10.13 sqa-env
     pyenv activate sqa-env
     ```

   [option 2: with conda]
   - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if not already installed.
   - Create a new conda environment (e.g. Python 3.10):
     ```sh
     conda create -n sqa-env python=3.10
     conda activate sqa-env
     ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **API Keys & Configuration**
   - create a `.env` file
   - copy content from `.env.example` into the `.env`
   - put your keys there. The `.env` file will not get tracked by git.
   - external API endpoints or urls are configured in `env_settings.yaml`.

5. **Run the project**
   ```sh
   python main.py
   ```

## Logging

All actions, results, and errors are logged automatically during workflow execution. Log files are stored in the `logs/` directory. You can review these logs for troubleshooting, auditing, or monitoring purposes.

## Linting and Code Style

Project uses [flake8](https://flake8.pycqa.org/) for Python linting and code style checks. To check your code:

```sh
pip install flake8
flake8 . # for whole project
flake8 <file_name> # for specific file
```

## Collaboration & Git Workflow

To ensure smooth collaboration, please follow this simplified git workflow:

1. **Clone the project**
   ```sh
   git clone git@git.ouryahoo.com:CAKE-AI/SQA.git
   cd <project-directory>
   ```
2. **Create a branch**
   - For JIRA-related work, name your branch after the ticket (e.g., `JK/CAKE-6218`).
   - Each branch should relate to only one ticket.
   ```sh
   git branch JK/CAKE-6218
   git checkout JK/CAKE-6218
   ```
3. **Make changes on your branch**
   - Stage files individually for clarity:
     ```sh
     git add <file1> <file2>
     ```
   - Only use `git add .` if you are certain all untracked files should be included.
   - Commit with a clear message:
     ```sh
     git commit -m "Describe your changes"
     ```
4. **Keep your branch up to date**
   - Regularly rebase from `main`:
     ```sh
     git checkout main
     git pull
     git checkout CAKE-6218
     git rebase main
     ```
5. **Push your branch**
   - For the first push:
     ```sh
     git push -u origin CAKE-6218
     ```
   - For subsequent pushes:
     ```sh
     git push
     ```
