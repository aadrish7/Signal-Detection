import asyncio
import os
import sys
import json
from crawlee_parser import run_parser
from LLM import detect_signals_with_llm, DEFAULT_ENDPOINT_NAME

async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths in the same directory
    parser_output = os.path.join(current_dir, "crawlee_result.json")
    llm_output = os.path.join(current_dir, "signal_detection_output.json")
    prepared_input = os.path.join(current_dir, "prepared_input.json")
    raw_llm_response = os.path.join(current_dir, "raw_llm_response.json")

    print(f"--- Starting Orchestration for {url} ---")
    
    # 1. Run Parser
    print("\nStep 1: Running Crawlee Parser...")
    parser_result = await run_parser(url, output_filename=parser_output)
    
    if not parser_result:
        print("Error: Parser failed to extract data.")
        return

    # 2. Run LLM Detection
    print("\nStep 2: Running LLM Signal Detection...")
    try:
        llm_result = detect_signals_with_llm(
            input_path=parser_output,
            endpoint_name=DEFAULT_ENDPOINT_NAME,
            prepared_output_path=prepared_input,
            raw_llm_output_path=raw_llm_response,
            final_output_path=llm_output
        )
        
        print("\n--- ALL STEPS COMPLETE ---")
        print(f"Parser output: {parser_output}")
        print(f"LLM output: {llm_output}")
        print("\nFinal Result Preview:")
        print(json.dumps(llm_result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error during LLM detection: {e}")

if __name__ == "__main__":
    asyncio.run(main())
