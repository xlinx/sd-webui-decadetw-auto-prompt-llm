import launch

if not launch.is_installed("OpenAI"):
    launch.run_pip(f"install OpenAI", "OpenAI")
