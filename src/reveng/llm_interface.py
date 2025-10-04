import logging
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, Template
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)


class BaseLLMInterface:
    """Base class for LLM interactions with common functionality."""

    def __init__(
        self,
        model_name: str,
        template_path: Optional[Path | str] = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            template_path: Optional override for the Jinja prompt template.
            temperature: Forwarded to the model.
        """
        self.model_name = model_name
        self.temperature = temperature

        resolved_template = (
            Path(template_path)
            if template_path is not None
            else Path(__file__).parent / "templates" / "default_template.j2"
        )
        self._template = self._load_template(resolved_template)

    @staticmethod
    def _load_template(template_path: Path) -> Template:
        """Load a Jinja2 template from file."""
        env = Environment(
            loader=FileSystemLoader(template_path.parent),
        )
        return env.get_template(template_path.name)

    @retry(
        stop=stop_after_attempt(3),
        # 5-120 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=5, max=120),
        reraise=True,
    )
    def _completion_with_retry(
        self,
        **kwargs,
    ):
        """Make LLM completion request with retry logic."""
        response = completion(
            **kwargs,
        )
        return response

    def _make_completion_request(
        self,
        prompt: str,
        **kwargs,
    ):
        """Make a completion request with standard parameters."""
        try:
            response = self._completion_with_retry(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                **kwargs,
            )
            return response
        except Exception as exc:
            logger.error(f"Model request failed: {exc}")
            raise

    def render_template(self, **kwargs) -> str:
        """Render the template with given context."""
        return self._template.render(**kwargs)
