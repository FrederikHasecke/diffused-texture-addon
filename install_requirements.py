import pip
import os
import subprocess
import sys


def install_requirements(requirements_file):
    """Install packages listed in the requirements.txt file using pip."""
    try:
        if os.path.exists(requirements_file):
            # Install packages from the requirements.txt file
            pip.main(["install", "-r", requirements_file])
            print("All libraries installed successfully.")
        else:
            print(f"The file {requirements_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Path to your requirements.txt file
requirements_path = (
    r"C:\Users\fredd\Desktop\SD-texturing\code\diffused_texture_addon\requirements.txt"
)

# Install other requirements
install_requirements(requirements_path)
