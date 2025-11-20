# Contributing to softadapt

Thank you for considering contributing to the softadapt package! Your contributions are invaluable to improve this project. Please review the following guidelines before submitting changes.

## How to Contribute
1. **Fork the Repository**: Fork the softadapt repository.
2. **Set Up the Environment**:
   ```bash
   git clone ssh://git@gitlab.com:<your-project>/SoftAdapt-Keras.git
   cd SoftAdapt-Keras
   uv sync
   ```
3. **Code Style**: Follow PEP8 guidelines and use `ruff` for formatting.
    To have code automatically checked and formatted before commit install the `pre-commit` rules.
   ```bash
   uv run pre-commit install
   ```
4. **Testing**: Run tests with `pytest` before submitting changes.
    ```bash
    KERAS_BACKEND=torch uv run --with pytest pytest
    ```
5. **Commit Styles**: Commits must follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format, checked with a `pre-commit` rule. For ease of use, use the `cz` command:
    1. Tool installation:
        ```bash
        uv tool install commitizen
        ```
    2. Usage
        ```bash
        cz c
        ```
        or 
        ```bash
        cz c -- -S
        ```
        to sign a commit.
5. **Submit Pull Request**: Push your changes to your fork and create a pull request.

## Submission Guidelines
- Include tests for new features
    - Minimum of 60% code coverage
    - Changes should not break existing tests
- Include docstrings for all new code
    - Project wiki is built from docstrings
- If additional documentation needed, provide delineated markdown in the pull request for transfer to the Wiki.
- Include a brief description of your changes in the pull request in a format that is consistent with [CHANGELOG.md](CHANGELOG.md).

For more details, see the [README.md](README.md) file.