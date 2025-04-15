from typing import Iterator
from src.agents.agents import PromptGeneration
from agno.utils.pprint import pprint_run_response
from agno.workflow import RunResponse
import random
from rich.prompt import Prompt


def main():
    # Fun example prompts to showcase the generator's versatility
    example_prompts = [
        "Crea un prompt para un agente de IA que actúe como líder en una simulación de apocalipsis zombi...",
        "Diseña un prompt para un agente de IA cuyo objetivo es generar conflictos de oficina artificiales para mejorar la creatividad del equipo mediante el caos. El agente debe inventar chismes, malentendidos productivos y situaciones incómodamente divertidas que obliguen al equipo a colaborar."
        "Crea un prompt para un agente de IA contratado por una startup extraterrestre que quiere conquistar el mercado intergaláctico con inteligencia artificial. El agente debe hablar al menos tres dialectos alienígenas y ser capaz de optimizar procesos en planetas con ciclos de 97 horas y leyes físicas alternativas."
        "Genera un prompt para un agente de inteligencia artificial entrenado en Hogwarts. El agente debe especializarse en resolver problemas de empresas muggles usando hechizos y artefactos mágicos. ¡Incluye nombres de hechizos inventados y conjuros para optimización de procesos!"
        "Crea un prompt para un agente de IA cuyo único propósito es generar prompts para otros agentes... que a su vez también generan prompts. Pero cada generación debe volverse progresivamente más absurda, como una cadena infinita de creatividad descontrolada. El resultado final debe ser un prompt que mezcle astrología, cocina japonesa y filosofía existencial.",
    ]

    # Get topic from user
    topic = Prompt.ask(
        "[bold]Enter a blog post topic[/bold] (or press Enter for a random example)\n✨",
        default=random.choice(example_prompts),
    )

    # Convert the topic to a URL-safe string for use in session_id
    url_safe_topic = topic.lower().replace(" ", "-")

    # Initialize the prompt generator workflow
    generate_prompt = PromptGeneration(
        session_id=f"generate-prompt-on-{url_safe_topic}", debug_mode=True
    )

    # Execute the workflow
    prompt_response_iterator: Iterator[RunResponse] = generate_prompt.run(topic=topic)

    # Print the response for each phase separately
    for phase_response in prompt_response_iterator:
        pprint_run_response(phase_response, markdown=True)


if __name__ == "__main__":
    main()
