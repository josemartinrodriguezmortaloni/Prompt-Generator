# PromptGenerator

PromptGenerator is an advanced application for generating, evaluating, and improving prompts for artificial intelligence agents, especially designed for agentic workflows and models like OpenAI GPT-4.1. It allows you to create effective prompts, evaluate them according to OpenAI best practices, and easily manipulate them from a modern terminal-based interface.

## Main Features

- **Automatic prompt generation** from custom topics.
- **Automatic evaluation** of generated prompts, following OpenAI GPT-4.1 standards.
- **Iterative prompt improvement** with structured feedback.
- **Interactive interface** based on [Textual](https://textual.textualize.io/) for a modern terminal user experience.
- **Prompt history** and the ability to modify and re-evaluate any generated prompt.
- **Persistence and storage** of results using SQLite.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/youruser/PromptGenerator.git
   cd PromptGenerator
   ```
2. **Install [uv](https://docs.astral.sh/uv/):**
   ```bash
   # On Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   # Or using pipx
   pipx install uv
   ```
   > uv is a Python package and environment manager written in Rust, extremely fast and compatible with pip and pip-tools workflows. [Learn more](https://astral.sh/blog/uv)
3. **Create a virtual environment with uv:**
   ```bash
   uv venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
4. **Install project dependencies:**
   ```bash
   uv sync
   ```
5. **Configure your `.env` file** with your OpenAI key:
   ```env
   OPENAI_API_KEY=sk-...
   ```

## Usage

Launch the main interface with:

```bash
uv run .
```

- Enter the topic for the prompt you want to generate.
- Click "Generate Prompt" to create and evaluate a new prompt.
- Modify the last generated prompt and re-evaluate it with "Modify Last".
- Check the prompt and evaluation history at the bottom of the interface.

## Main Libraries Used

- **[agno](https://pypi.org/project/agno/):** Framework for agentic workflows, agent integration, tools, and persistent storage.
- **[openai](https://pypi.org/project/openai/):** Official client for interacting with the OpenAI API, including GPT-4.1 models and agent tools.
- **[rich](https://rich.readthedocs.io/):** Library for advanced text formatting and logging in the terminal, used for attractive result and log visualization.
- **[sqlalchemy](https://www.sqlalchemy.org/):** ORM toolkit for persistence and manipulation of SQLite databases.
- **[textual](https://textual.textualize.io/):** Framework for building modern, reactive terminal user interfaces using CSS and widgets.
- **[uv](https://docs.astral.sh/uv/):** Extremely fast Python package and project manager, written in Rust. Replaces pip, pip-tools, pipx, poetry, virtualenv, and more. See the [official documentation](https://docs.astral.sh/uv/) for details.

## Example Workflow

1. The user enters a topic (e.g., "AI for space exploration").
2. The agent generates an initial prompt, evaluates it, and improves it using structured feedback.
3. The final prompt is automatically saved and can be modified and re-evaluated.
4. The entire process and results are displayed in real time in the interface.

## Customization and Development

- You can modify the base prompts and agent instructions in `src/agents/agents.py`.
- The interface can be customized by editing the CSS and widgets in `__main__.py`.
- For more information on writing your own `pyproject.toml` and managing dependencies, see the [official Python Packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

## License

This project is distributed under the MIT license. See the `LICENSE` file for more details.

---

> **References:**
>
> - [OpenAI GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
> - [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
> - [Textual Documentation](https://textual.textualize.io/)
> - [uv Documentation (Astral)](https://docs.astral.sh/uv/)
