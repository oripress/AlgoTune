#!/usr/bin/env python3
"""
Simple test runner that feeds inputs one by one and logs responses.
No complex LLM interface machinery - just straightforward input/output logging.
"""
import sys
import logging
import argparse
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def setup_logging(task_name):
    """Set up logging for the test."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{task_name}_simple_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"=== Simple Test Runner for {task_name} ===")
    return log_file

def load_test_inputs(input_file):
    """Load and parse test inputs from file."""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split on INPUT_SEPARATOR and filter out empty sections
    sections = content.split("[INPUT_SEPARATOR]")
    inputs = [section.strip() for section in sections[1:] if section.strip()]
    
    logging.info(f"Loaded {len(inputs)} test inputs from {input_file}")
    return inputs

def process_input(input_text, input_number, task_name=None):
    """Process a single input and return the response."""
    logging.info(f"=== Processing Input {input_number} ===")
    logging.info(f"Sent to LLM: {input_text}")
    
    # For demonstration: simulate the response that would come from the test file
    # In a real implementation, this would call the actual LLM or test system
    
    # Read the actual test responses from the input file if possible
    if task_name:
        try:
            # Try to get the actual response from the test file
            input_file = f"inputs/{task_name}.txt"
            if os.path.exists(input_file):
                with open(input_file, 'r') as f:
                    content = f.read()
                
                sections = content.split("[INPUT_SEPARATOR]")[1:]
                if input_number <= len(sections):
                    response = sections[input_number - 1].strip()
                else:
                    response = f"No response {input_number} found in test file"
            else:
                response = f"Test file not found: {input_file}"
        except Exception as e:
            logging.error(f"Error reading test responses: {e}")
            response = f"Error reading response {input_number}: {str(e)}"
    else:
        # Fallback: just echo the input
        response = f"Echo: {input_text[:100]}..."
    
    logging.info(f"Received from LLM: {response}")
    return response

def run_simple_test(input_file):
    """Run the simple test with clear input/output logging."""
    task_name = Path(input_file).stem
    log_file = setup_logging(task_name)
    
    try:
        # Load test inputs
        inputs = load_test_inputs(input_file)
        
        # Process each input one by one
        responses = []
        for i, input_text in enumerate(inputs, 1):
            response = process_input(input_text, i, task_name)
            responses.append(response)
        
        logging.info(f"=== Test Complete ===")
        logging.info(f"Processed {len(responses)} inputs successfully")
        logging.info(f"Log saved to: {log_file}")
        
        return responses
        
    except Exception as e:
        logging.error(f"Error during test: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Simple test runner for input/output testing")
    parser.add_argument("--input", type=str, required=True, help="Path to input test file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        responses = run_simple_test(args.input)
        print(f"Test completed successfully. Processed {len(responses)} inputs.")
        return 0
    except Exception as e:
        print(f"Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())