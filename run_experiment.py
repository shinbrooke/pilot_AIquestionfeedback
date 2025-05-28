#!/usr/bin/env python
"""
Launch script for the Bloom's Taxonomy Question Study.
This script provides a convenient way to start the experiment with various options.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from dotenv import load_dotenv
import time

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import langchain
        import openai
        
        # Try to import psychopy but don't fail if not available
        try:
            import psychopy
            print("✓ PsychoPy is available for parallel port functionality")
        except ImportError:
            print("⚠ PsychoPy not found - parallel port functionality will be disabled")
        
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install all dependencies with: pip install -r requirements.txt")
        return False

def check_openai_key():
    """Check if OpenAI API key is available."""
    # Try to load from .env file
    load_dotenv()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Warning: OpenAI API key not found in environment variables or .env file")
        print("The experiment will fail when generating AI feedback unless you provide a key")
        
        set_key = input("Would you like to enter an OpenAI API key now? (y/n): ")
        if set_key.lower() == 'y':
            key = input("Enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = key
            
            # Also save to .env file for future use
            with open(".env", "a") as f:
                f.write(f"\nOPENAI_API_KEY={key}\n")
            
            print("✓ API key saved to environment and .env file")
        else:
            return False
    else:
        print("✓ OpenAI API key found")
    
    return True

def configure_paragraphs():
    """Configure experiment paragraphs if needed."""
    if not os.path.exists("paragraphs_config.py"):
        print("⚠ Warning: paragraphs_config.py not found")
        create_config = input("Would you like to create a template paragraphs_config.py file? (y/n): ")
        
        if create_config.lower() == 'y':
            with open("paragraphs_config.py", "w") as f:
                f.write('''"""
Configuration file for experiment paragraphs.
Replace these sample paragraphs with your actual stimuli before running the experiment.
"""

PARAGRAPHS = [
    "Climate change is altering ecosystems worldwide. Rising temperatures are causing glaciers to melt, sea levels to rise, and weather patterns to become more extreme. These changes are affecting both natural habitats and human communities, with some regions experiencing drought while others face increased flooding.",
    
    "The human brain contains approximately 86 billion neurons. These neurons communicate through electrical and chemical signals, forming complex networks that enable consciousness, cognition, and behavior. Modern neuroscience continues to uncover how these neural connections give rise to our thoughts and experiences.",
    
    # Add the remaining paragraphs here
]

# You should have 40 paragraphs total for the experiment
# If you have fewer than 40, the remaining will be filled with placeholders
def get_paragraphs(count=40):
    """Return the specified number of paragraphs, filling with placeholders if needed."""
    result = PARAGRAPHS.copy()
    
    # If we need more paragraphs than provided, add placeholders
    if len(result) < count:
        for i in range(len(result), count):
            result.append(f"Sample paragraph {i+1}. Replace this with your actual experimental stimulus.")
    
    return result[:count]
''')
            print("✓ Created paragraphs_config.py template")
            print("⚠ Remember to edit this file to include your experiment paragraphs")
            
            edit_now = input("Would you like to edit the file now? (y/n): ")
            if edit_now.lower() == 'y':
                # Try to open with default editor
                try:
                    if sys.platform == 'win32':
                        os.system("notepad paragraphs_config.py")
                    elif sys.platform == 'darwin':  # macOS
                        os.system("open -e paragraphs_config.py")
                    else:  # Linux and other Unix
                        editors = ["nano", "vim", "emacs", "gedit"]
                        for editor in editors:
                            if subprocess.call(["which", editor], stdout=subprocess.PIPE) == 0:
                                os.system(f"{editor} paragraphs_config.py")
                                break
                except Exception as e:
                    print(f"Could not open editor: {e}")
                    print("Please manually edit paragraphs_config.py before running the experiment")
    else:
        # Check if there are enough paragraphs defined
        try:
            from paragraphs_config import get_paragraphs
            paragraphs = get_paragraphs(40)
            print(f"✓ Found {len(paragraphs)} paragraphs in configuration")
            
            if len(paragraphs) < 40:
                print("⚠ Warning: Fewer than 40 unique paragraphs defined. Placeholders will be used.")
        except Exception as e:
            print(f"⚠ Warning: Error in paragraphs_config.py: {e}")
            return False
    
    return True

def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Run the Bloom's Taxonomy Question Study experiment")
    
    parser.add_argument('--port', type=int, default=8501,
                        help='Port to run Streamlit on (default: 8501)')
    
    parser.add_argument('--no-browser', action='store_true',
                        help="Don't open the browser automatically")
    
    parser.add_argument('--test-parallel', action='store_true',
                        help='Run a test of the parallel port before starting')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("\n" + "="*70)
    print("Bloom's Taxonomy Question Study - Experiment Launcher")
    print("="*70 + "\n")
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Check OpenAI API key
    print("\nChecking OpenAI API key...")
    check_openai_key()
    
    # Configure paragraphs
    print("\nChecking experiment paragraphs...")
    configure_paragraphs()
    
    # Test parallel port if requested
    if args.test_parallel:
        print("\nTesting parallel port...")
        try:
            from parallel_port import ParallelPortHandler
            port_handler = ParallelPortHandler()
            
            if port_handler.available:
                print("Sending test markers to parallel port...")
                port_handler.test_markers()
                print("✓ Parallel port test completed")
            else:
                print("⚠ Parallel port functionality not available")
                print("The experiment will run but won't send hardware markers")
        except Exception as e:
            print(f"Error testing parallel port: {e}")
            print("⚠ Parallel port test failed, but the experiment can still run")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("\n✓ Created logs directory")
    
    # Start the Streamlit app
    print("\nStarting experiment application...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port)]
    
    # Add additional Streamlit arguments to reduce verbosity
    cmd.extend(["--server.headless", "true", "--logger.level", "error"])
    
    # Open browser if requested
    if not args.no_browser:
        time.sleep(1)  # Give the server a moment to start
        webbrowser.open(f"http://localhost:{args.port}")
    
    # Run the Streamlit process
    print(f"\n✓ Experiment running at http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the experiment\n")
    
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nExperiment stopped by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())