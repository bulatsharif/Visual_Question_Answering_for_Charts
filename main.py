"""
Main program entrypoint for the Visual Question Answering for Charts project.

This very small module provides a simple `main()` stub used by tools
or CI to ensure the package can be invoked as a script. It intentionally
does not implement program logic; the `chartvqa` package contains the
actual training and evaluation CLI tools.
"""


def main():
    """Simple runner used as an entrypoint for package-level execution.

    This prints a confirmation string and returns. Keep the implementation
    intentionally small to avoid side effects (e.g., GPU allocation) when
    importing the package.
    """
    print("Main function executed.")


if __name__ == "__main__":
    main()
