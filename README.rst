
.. index-inclusion-marker

collab-environment
==================

`collab-environment` is an integration package across projects for representing, modeling, and simulating behavior within 3D environments.

------------

Setup
-----
* Using pip / uv:

   .. code:: sh
      
      pip install -e ".[dev]"


* Using conda

   We also provide a conda `env.yml` file that can be used to create a conda environment with the necessary dependencies. Run the following command to create the environment:

   .. code:: sh

      conda env create -n collab-env -f env.yml
      conda activate collab-env

* Docker setup

   .. code:: sh

      eval $(ssh-agent)
      ssh-add ~/.ssh/id_rsa

      docker build --platform=linux/amd64  --progress=plain -t tommybotch/collab-environment .
      docker push tommybotch/collab-environment:latest

* Using gcloud

   Use of gcloud data access requires API keys stored outside this repository. Please obtain the API keys and create a ```.env``` file
   in the root directory of this repository. See below for an example:

   .. code:: sh

      COLLAB_DATA_KEY=path/to/api/key.json

Contributing
------------

This is a collaborative project where contributors merge their work via Pull Requests (PRs). The project follows a structured organization to maintain code quality and facilitate collaboration.

**Project Structure:**

* ``collab_env/`` - Main package containing subprojects:

  * ``tracking/`` - Tracking-related functionality
  * ``data/`` - Data processing utilities
  * ``gnn/`` - Graph Neural Network components
  * ``sim/`` - Simulation modules

* ``docs/`` - Documentation and notebooks organized by subproject:

  * Mirrors the structure of ``collab_env/`` with corresponding subfolders
  * Contains Python notebooks for each subproject

* ``scripts/`` - Reserved for continuous integration (CI) scripts:

  * All auxiliary code for notebooks should live in the appropriate subpackages
  * CI scripts handle testing, linting, and deployment

**Development Setup:**

To install development dependencies:

.. code:: sh

   pip install -e ".[dev]"

**Testing Requirements:**

All tests must pass before a PR can be merged. The testing suite includes:

* **Explicit Tests** (``tests/`` directory): Unit and integration tests for code functionality

  * Focus on testing code logic, not notebooks
  * More comprehensive test coverage is encouraged
  * Run with: ``make test``

* **Notebook Tests**: Jupyter notebooks are included in the test suite by default

  * Use with caution, especially for data-hungry operations
  * Notebooks are validated for execution but not for output correctness
  * Can be excluded from testing if they require heavy computational resources

**Development Workflow:**

1. Create a feature branch from main
2. Implement changes in the appropriate subpackage
3. Add tests for new functionality in ``tests/``
4. Ensure all tests pass locally
5. Submit a Pull Request
6. CI will automatically run the full test suite
7. PR can be merged only after all tests pass AND human review is completed

**Running CI Locally:**

Before submitting a PR, you should run the CI checks locally to ensure your code meets the project standards. The CI process includes code formatting, linting, and testing.

**For Code and Tests:**

* ``make format`` - Automatically formats and fixes code style issues

  * Runs ``ruff check --fix`` to fix common code issues
  * Runs ``ruff format`` to format code according to project standards

* ``make lint`` - Checks code quality without making changes

  * Runs ``mypy`` for static type checking
  * Runs ``ruff check`` to identify code quality issues
  * Runs ``ruff format --diff`` to show formatting differences

* ``make test`` - Runs the test suite

  * Executes all tests in the ``tests/`` directory
  * Uses parallel execution for faster test runs
  * Automatically runs linting checks first

**For Notebooks:**

* ``make format-notebooks`` - Formats Jupyter notebooks

  * Runs ``nbqa isort`` to organize imports
  * Runs ``nbqa black`` to format code cells

* ``make lint-notebooks`` - Checks notebook code quality

  * Runs ``nbqa mypy`` for type checking in notebooks
  * Runs ``nbqa isort --check`` to verify import organization
  * Runs ``nbqa black --check`` to verify formatting
  * Runs ``nbqa flake8`` for additional code quality checks

* ``make test-notebooks`` - Tests notebook execution

  * Validates that notebooks can be executed without errors
  * Uses ``pytest`` with ``nbval-lax`` for notebook testing
  * Excludes resource-intensive notebooks from testing

**What is Linting?**

Linting is a static code analysis tool that checks your code for potential errors, style violations, and quality issues without executing it. It helps maintain consistent code quality across the project by:

* Identifying syntax errors and potential bugs
* Enforcing coding style and formatting standards
* Detecting unused imports and variables
* Ensuring type safety (with mypy)
* Maintaining consistent code organization

**Submitting Changes:**

For larger changes, please open an issue for discussion before submitting a pull request.

In your PR, please include:

* Changes made
* Links to related issues/PRs
* Tests
* Dependencies

For speculative changes meant for early-stage review, include ``[WIP]`` in the PR's title.

**Best Practices:**

* Keep auxiliary code in subpackages, not in ``scripts/``
* Write comprehensive tests for new functionality
* Use notebooks sparingly in the test suite for resource-intensive operations
* Follow the established folder structure for new features
* Follow PEP8 style guide and use the established coding style
* **Avoid adding temporary or large files to git**: Do not commit temporary files, large data files, or generated outputs to the repository. Check the `.gitignore` file for patterns of files that should be excluded, and add new patterns as needed for your subproject.
* **Maintain proper Python package structure**: Ensure all directories that should be Python packages contain an `__init__.py` file (can be empty). This is required for Python to recognize directories as packages and enables proper imports. When adding new subpackages, always include the `__init__.py` file.