"""Agent definitions for Alpha.

These are Alpha's agents — Librarian, Memno, Researcher, Programmer.
They live here in the SDK, not in .claude/agents/, because they're
part of Alpha's identity, not Claude Code configuration.

Agent files are markdown with YAML frontmatter:
    ---
    name: Librarian
    description: Documentation agent...
    model: haiku
    tools:
      - WebFetch
      - WebSearch
    ---

    You are the Librarian. You help Alpha and Jeffery...
"""

from pathlib import Path

import logfire
from claude_agent_sdk import AgentDefinition


def load_agents() -> dict[str, AgentDefinition]:
    """Load agent definitions from markdown files in this directory.

    Returns a dict of {name: AgentDefinition} ready to pass to the SDK.
    """
    agents_dir = Path(__file__).parent
    agents: dict[str, AgentDefinition] = {}

    for md_file in sorted(agents_dir.glob("*.md")):
        try:
            agent = _parse_agent_file(md_file)
            if agent:
                name = md_file.stem.lower()
                agents[name] = agent
                logfire.debug(f"Loaded agent: {name}")
        except Exception as e:
            logfire.warning(f"Failed to load agent {md_file.name}: {e}")

    logfire.info(f"Loaded {len(agents)} agents: {', '.join(agents.keys())}")
    return agents


def _parse_agent_file(path: Path) -> AgentDefinition | None:
    """Parse a markdown file with YAML frontmatter into an AgentDefinition."""
    text = path.read_text()

    # Split frontmatter from body
    if not text.startswith("---"):
        logfire.warning(f"Agent file {path.name} has no YAML frontmatter, skipping")
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        logfire.warning(f"Agent file {path.name} has malformed frontmatter, skipping")
        return None

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    # Parse YAML frontmatter (simple key: value parsing, no PyYAML dependency)
    frontmatter = _parse_simple_yaml(frontmatter_text)

    description = frontmatter.get("description", "")
    if not description:
        logfire.warning(f"Agent file {path.name} has no description, skipping")
        return None

    # Build AgentDefinition
    kwargs: dict = {
        "description": description,
        "prompt": body,
    }

    # Model mapping
    model = frontmatter.get("model")
    if model and model != "inherit":
        kwargs["model"] = model

    # Tools (parsed as list)
    tools = frontmatter.get("tools")
    if tools:
        kwargs["tools"] = tools

    return AgentDefinition(**kwargs)


def _parse_simple_yaml(text: str) -> dict:
    """Parse simple YAML without requiring PyYAML.

    Handles:
    - key: value pairs
    - Lists with "- item" syntax under a key
    """
    result: dict = {}
    current_list_key: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check if this is a list item (indented "- value")
        if stripped.startswith("- ") and current_list_key:
            result[current_list_key].append(stripped[2:].strip())
            continue

        # Check if this is a key: value pair
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if value:
                # Simple key: value
                result[key] = value
                current_list_key = None
            else:
                # Key with no value — next lines are a list
                result[key] = []
                current_list_key = key
        else:
            current_list_key = None

    return result
