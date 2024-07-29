import sys
import subprocess
import logging

def install_requirements(force=False) -> None:
    """
    Invoke pip to install the requirements for the extension.
    """
    try:
        from launch import args

        if getattr(args, "skip_install", False):
            logging.info(
                "webui launch.args.skip_install is true, skipping sd-webui-decadetw-auto-prompt-llm",
            )
            return
    except ImportError:
        pass

    requirements_to_install = [
        "selenium", "OpenAI"
    ]

    if not requirements_to_install:
        return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *requirements_to_install,
    ]
    print(f"sd-webui-decadetw-auto-prompt-llm installer: running {' '.join(command)}")
    subprocess.check_call(command)

if __name__ == "__main__":

    install_requirements(force=("-f" in sys.argv))