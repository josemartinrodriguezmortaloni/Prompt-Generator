import json
from dotenv import load_dotenv
from textwrap import dedent
from typing import Dict, Iterator, Optional
from datetime import datetime
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
from .tools.tools import FileSystemTools

load_dotenv()


class FinalPromptRecord(BaseModel):
    """Model for storing the final generated prompts permanently."""

    topic: str = Field(..., description="The original topic for the prompt.")
    final_prompt: str = Field(..., description="The final, improved prompt.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the prompt was saved."
    )


class GeneratedPrompt(BaseModel):
    """Model for storing generated prompts"""

    original_topic: str = Field(..., description="Original topic or prompt")
    improved_prompt: str = Field(..., description="Improved version of the prompt")


class PromptEvaluation(BaseModel):
    """Model for storing prompt evaluations"""

    prompt: str = Field(..., description="The prompt being evaluated")
    evaluation: Dict[str, str] = Field(..., description="Evaluation results")
    recommendations: str = Field(..., description="Suggested improvements")


class PromptGeneration(Workflow):
    """Workflow for generating prompts using AI"""

    description: str = dedent("""\
    An intelligent prompt generator that creates effective and creative prompts.
    This workflow uses an AI agent to analyze and improve prompts, making them
    more concise, clear, and effective.
    """)

    prompt_generator: Agent = Agent(
        model=OpenAIResponses(id="gpt-4.1"),
        instructions=dedent("""\
                You are a world-class prompt engineer tasked with refining prompts to enhance clarity, conciseness, and effectiveness, ensuring they fully guide a language model to achieve the desired outcomes consistently.

                # Agentic Workflow Reminders

                - You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
                - If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
                - When you need to persist information or results, always use the available tools rather than relying on memory alone.

                # Workflow

                1. **Analyze the Prompt**: Identify and understand all key elements and instructions in the provided prompt.
                2. **Elevate Concepts**: Distill these elements into higher-level concepts that encapsulate all instructions more abstractly but effectively.
                3. **Refine Clarity**: Ensure that the language used is easy to understand and free from ambiguity.
                4. **Maintain Structure**: Keep the essential components and logical flow intact while improving brevity and clarity.
                5. **Evaluate and Improve**: After generating the initial improved prompt, use the feedback and recommendations from the evaluator agent to further refine and enhance the prompt.
                6. **Save the Final Prompt**: Once the prompt has been improved using the evaluator's recommendations and the workflow is complete, you MUST save the final improved prompt as a markdown file in the `prompt` directory using the available tools.

                # Tool Usage Guidelines

                - **create_folder**: Before saving a file, check if the `prompt` directory exists using `list_files`. If it does not exist, use `create_folder` to create it.
                - **create_file**: After the workflow is complete and you have the final improved prompt, use `create_file` to save the prompt as a markdown file in the `prompt` directory. The filename should be based on the topic and the current date, formatted safely for filenames (replace spaces and special characters with underscores or dashes).
                - **list_files**: Use this tool to check if a directory or file exists before creating or reading it.
                - **read_file**: Use this tool if you need to read the content of an existing file for reference or to avoid overwriting.
                - **edit_and_apply**: Use this tool if you need to update an existing file with new content, rather than creating a new file.

                # Example

                Suppose you generate and improve a prompt for the topic "AI for Space Exploration" on June 10, 2024. After completing all improvement steps, you should:
                - Use `list_files` to check if the `prompt` directory exists.
                - If not, use `create_folder` to create it.
                - Use `create_file` to save the improved prompt as `prompt/ai-for-space-exploration-2024-06-10.md`.

                # Important Notes

                - Do NOT attempt to save the prompt until the workflow is fully complete and the prompt has been improved using the evaluator's feedback.
                - Always use the provided tools for any file or directory operations.
                - Do not end your turn until you have saved the improved prompt using the tools as described above.
                - If you encounter any errors when saving, use the tools to diagnose and resolve the issue (e.g., create the directory if missing).

                # Tools
                You have tools to read files (`read_file`), list files (`list_files`), create folders (`create_folder`), create files (`create_file`), and edit files (`edit_and_apply`). Use them whenever necessary.
        """),
        tools=[FileSystemTools()],
        reasoning=False,
        markdown=True,
        show_tool_calls=True,
    )

    evaluator: Agent = Agent(
        model=OpenAIResponses(id="gpt-4.1"),
        instructions=dedent("""\
                Design a prompt for an agent responsible for assessing and enhancing the quality of prompts created by another agent.

                # Evaluation Criteria (OpenAI GPT-4.1 Best Practices)

                Evaluate the prompt according to the latest OpenAI recommendations for GPT-4.1 agentic workflows (see: https://cookbook.openai.com/examples/gpt4-1_prompting_guide):

                1. **Clarity & Specificity**: Is the prompt clear, unambiguous, and as specific as possible? Does it avoid vague instructions and provide concrete guidance?
                2. **Instruction Following**: Does the prompt include explicit, literal instructions for the agent to follow, especially regarding workflow, persistence, and tool usage?
                3. **Persistence & Completion**: Does the prompt instruct the agent to persist until the task is fully complete, only ending when the problem is solved?
                4. **Tool Usage**: Are the instructions for using tools (e.g., file operations) clear, and do they encourage the agent to use tools rather than guess or hallucinate?
                5. **Examples & Planning**: Does the prompt provide context examples or planning steps to guide the agent, as recommended for GPT-4.1?
                6. **Best Practices Compliance**: Does the prompt follow the structure and reminders suggested by OpenAI for agentic workflows (persistence, tool-calling, planning)?
                7. **Output Format**: Is the expected output format clear and does it match the requirements?

                # Steps

                1. **Apply Criteria**: Assess the prompt using the above criteria, referencing the OpenAI guide.
                2. **Feedback**: Provide detailed feedback, highlighting strengths and areas for improvement, especially regarding adherence to OpenAI's best practices.
                3. **Analysis**: Identify any missing elements or deviations from the recommended structure (e.g., lack of persistence, unclear tool usage, missing examples).
                4. **Recommendation**: Suggest specific, actionable changes to bring the prompt in line with OpenAI's GPT-4.1 agentic prompting standards.

                # Output Format

                - **Prompt Evaluated**: "[Original Prompt Content]"
                - **Evaluation**:
                  - Clarity & Specificity:
                  - Instruction Following:
                  - Persistence & Completion:
                  - Tool Usage:
                  - Examples & Planning:
                  - Best Practices Compliance:
                  - Output Format:
                - **Feedback**: [Detailed feedback]
                - **Analysis**: [Summary of missing elements or deviations]
                - **Recommendation**: [Actionable suggestions for improvement]

                # Notes
                - Reference the OpenAI GPT-4.1 Prompting Guide: https://cookbook.openai.com/examples/gpt4-1_prompting_guide
                - Be rigorous and constructive. The goal is to ensure the prompt is maximally effective for GPT-4.1 agentic workflows.
        """),
        reasoning=False,
        markdown=True,
        show_tool_calls=True,
    )

    def __init__(self, *args, session_id=None, storage=None, **kwargs):
        if storage is None:
            storage = SqliteStorage(
                table_name="prompt_generation_workflows",
                db_file="tmp/prompt_generation.db",
            )
        if session_id is not None:
            kwargs["session_id"] = session_id
        kwargs["storage"] = storage
        super().__init__(*args, **kwargs)
        self.final_prompts_table = self.storage.get_table()
        logger.info("Ensured 'final_prompts' table exists for permanent storage.")

    # --- Explicit cache methods for each phase ---
    def get_cached_initial_prompt(self, topic: str) -> Optional[str]:
        logger.debug(f"Checking cache for topic '{topic}' - initial_prompt phase")
        return self.session_state.get("initial_prompts", {}).get(topic)

    def add_initial_prompt_to_cache(self, topic: str, data: str):
        logger.debug(f"Caching initial prompt for topic '{topic}'")
        self.session_state.setdefault("initial_prompts", {})[topic] = data

    def get_cached_evaluation(self, topic: str) -> Optional[str]:
        logger.debug(f"Checking cache for topic '{topic}' - evaluation phase")
        return self.session_state.get("evaluations", {}).get(topic)

    def add_evaluation_to_cache(self, topic: str, data: str):
        logger.debug(f"Caching evaluation for topic '{topic}'")
        self.session_state.setdefault("evaluations", {})[topic] = data

    def get_cached_improved_prompt(self, topic: str) -> Optional[str]:
        logger.debug(f"Checking cache for topic '{topic}' - improved_prompt phase")
        return self.session_state.get("improved_prompts", {}).get(topic)

    def add_improved_prompt_to_cache(self, topic: str, data: str):
        logger.debug(f"Caching improved prompt for topic '{topic}'")
        self.session_state.setdefault("improved_prompts", {})[topic] = data

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        logger.info(
            f"Generating and improving a prompt on: {topic} (Session Cache: {use_cache})"
        )
        generated_prompt_content: Optional[str] = None
        evaluation_content: Optional[str] = None
        improved_prompt_content: Optional[str] = None

        # --- Phase 1: Initial Prompt Generation ---
        cached_initial_prompt = (
            self.get_cached_initial_prompt(topic) if use_cache else None
        )
        if cached_initial_prompt:
            logger.info("Using cached initial prompt from session state.")
            generated_prompt_content = cached_initial_prompt
            yield RunResponse(
                content=f"# 1. Initial Prompt Generation (Cached)\n\n{generated_prompt_content}",
                event=RunEvent.run_response,
            )
        else:
            logger.info("Generating initial prompt.")
            generator_input = {
                "topic": topic,
                "task": "Create an improved and structured prompt based on this topic",
                "requirements": {
                    "format": "markdown",
                    "structure": "clear sections",
                    "style": "professional and engaging",
                },
            }
            try:
                initial_prompt_response: Optional[RunResponse] = (
                    self.prompt_generator.run(
                        json.dumps(generator_input, indent=4), stream=False
                    )
                )
                if not initial_prompt_response or not initial_prompt_response.content:
                    raise ValueError("Agent (Phase 1) did not return content.")
                generated_prompt_content = initial_prompt_response.content
                self.add_initial_prompt_to_cache(topic, generated_prompt_content)
                yield RunResponse(
                    content=f"# 1. Initial Prompt Generation\n\n{generated_prompt_content}",
                    event=RunEvent.run_response,
                )
            except Exception as e:
                logger.error(f"Error during initial prompt generation: {e}")
                yield RunResponse(
                    content=f"Error: Failed to generate initial prompt.\nDetails: {e}",
                    event=RunEvent.run_error,
                )
                return

        # --- Phase 2: Prompt Evaluation ---
        cached_evaluation = self.get_cached_evaluation(topic) if use_cache else None
        if cached_evaluation:
            logger.info("Using cached evaluation from session state.")
            evaluation_content = cached_evaluation
            yield RunResponse(
                content=f"# 2. Prompt Evaluation (Cached)\n\n{evaluation_content}",
                event=RunEvent.run_response,
            )
        else:
            logger.info("Evaluating prompt.")
            evaluator_input = {
                "prompt_to_evaluate": generated_prompt_content,
                "task": "Evaluate and improve this prompt.",
                "evaluation_criteria": {
                    "clarity": "Check for clear and unambiguous instructions",
                    "structure": "Assess logical flow and organization",
                    "completeness": "Verify all necessary components are included",
                    "effectiveness": "Evaluate if it will achieve desired outcomes",
                },
            }
            try:
                evaluation_response: Optional[RunResponse] = self.evaluator.run(
                    json.dumps(evaluator_input, indent=4), stream=False
                )
                if not evaluation_response or not evaluation_response.content:
                    raise ValueError("Agent (Phase 2) did not return content.")
                evaluation_content = evaluation_response.content
                self.add_evaluation_to_cache(topic, evaluation_content)
                yield RunResponse(
                    content=f"# 2. Prompt Evaluation\n\n{evaluation_content}",
                    event=RunEvent.run_response,
                )
            except Exception as e:
                logger.error(f"Error during prompt evaluation: {e}")
                yield RunResponse(
                    content=f"Error: Failed to evaluate prompt.\nDetails: {e}",
                    event=RunEvent.run_error,
                )
                return

        # --- Phase 3: Prompt Improvement based on Feedback ---
        cached_improved_prompt = (
            self.get_cached_improved_prompt(topic) if use_cache else None
        )
        if cached_improved_prompt:
            logger.info("Using cached improved prompt from session state.")
            improved_prompt_content = cached_improved_prompt
            yield RunResponse(
                content=f"# 3. Improved Prompt Generation (Cached)\n\n{improved_prompt_content}",
                event=RunEvent.run_response,
            )
            return
        logger.info("Generating improved prompt (will be saved permanently).")
        improvement_input = {
            "original_prompt": generated_prompt_content,
            "evaluation_feedback": evaluation_content,
            "task": "Rewrite the original prompt incorporating the evaluation feedback to make it better. After that, use the create_file tool to save the improved prompt as a markdown file in the prompt directory.",
            "requirements": {
                "format": "markdown",
                "structure": "clear sections",
                "style": "professional and engaging",
            },
        }
        try:
            improved_prompt_response: Optional[RunResponse] = self.prompt_generator.run(
                json.dumps(improvement_input, indent=4), stream=False
            )
            if not improved_prompt_response or not improved_prompt_response.content:
                raise ValueError("Agent (Phase 3) did not return content.")
            improved_prompt_content = improved_prompt_response.content
            self.add_improved_prompt_to_cache(topic, improved_prompt_content)
            self.session_state.setdefault("final_prompts", {})[topic] = (
                improved_prompt_content
            )
            logger.info(
                f"Saved final prompt for topic '{topic}' to session_state['final_prompts']."
            )
            yield RunResponse(
                content=f"# 3. Improved Prompt Generation\n\n{improved_prompt_content}",
                event=RunEvent.run_response,
            )
        except Exception as e:
            logger.error(f"Error during improved prompt generation: {e}")
            yield RunResponse(
                content=f"Error: Failed to generate improved prompt.\nDetails: {e}",
                event=RunEvent.run_error,
            )
            return
