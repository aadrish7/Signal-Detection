import asyncio
import csv
import os
import sys
from urllib.parse import urlparse
from crawlee_parser import run_parser
from LLM import detect_signals_with_llm, DEFAULT_ENDPOINT_NAME

def ensure_url(domain):
    if not domain.startswith(('http://', 'https://')):
        return 'https://' + domain
    return domain

async def main():
    input_csv = "leads.csv"
    output_csv = "leads_with_signals.csv"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, input_csv)
    output_path = os.path.join(current_dir, output_csv)
    
    parser_output = os.path.join(current_dir, "crawlee_result.json")
    prepared_input = os.path.join(current_dir, "prepared_input.json")
    raw_llm_response = os.path.join(current_dir, "raw_llm_response.json")
    llm_output = os.path.join(current_dir, "signal_detection_output.json")
    
    signals_keys = [
        "revenue_brand_signal",
        "engagement_ring_focus",
        "ecommerce_maturity",
        "ads_running",
        "multi_location_scale",
        "custom_ring_offering"
    ]

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    print(f"Starting batch processing from {input_path}")
    print(f"Output will be saved to {output_path}")

    # Count total rows for progress tracking
    with open(input_path, 'r', encoding='utf-8') as infile:
        total_rows = sum(1 for _ in infile) - 1 # Subtract header
    
    print(f"Total rows to process: {total_rows}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        
        # In the provided leads.csv, the column name is 'companyDomain', check for variations
        domain_key = "companyDomain"
        if domain_key not in reader.fieldnames:
            # Try lowercase or other variations just in case
            possible_keys = [k for k in reader.fieldnames if k.lower() == "companydomain" or k.lower() == "company_domain"]
            if possible_keys:
                domain_key = possible_keys[0]
            else:
                print("Error: 'companyDomain' column not found in CSV.")
                sys.exit(1)

        fieldnames = reader.fieldnames + signals_keys
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, row in enumerate(reader, 1):
            domain = row.get(domain_key, "").strip()
            if not domain:
                print(f"[{idx}/{total_rows}] Skipping empty domain.")
                for key in signals_keys:
                    row[key] = ""
                writer.writerow(row)
                continue
            
            url = ensure_url(domain)
            print(f"[{idx}/{total_rows}] Processing: {url}")
            
            # Default values in case of failure
            for key in signals_keys:
                row[key] = "Error"
            
            try:
                # 1. Run Parser
                parser_result = await run_parser(url, output_filename=parser_output)
                if parser_result:
                    # 2. Run LLM
                    llm_result = detect_signals_with_llm(
                        input_path=parser_output,
                        endpoint_name=DEFAULT_ENDPOINT_NAME,
                        prepared_output_path=prepared_input,
                        raw_llm_output_path=raw_llm_response,
                        final_output_path=llm_output
                    )
                    
                    extracted_signals = llm_result.get("signals", {})
                    for key in signals_keys:
                        signal_data = extracted_signals.get(key, {})
                        # Some results might be missing if LLM didn't format correctly
                        if isinstance(signal_data, dict):
                            detected = signal_data.get("detected", False)
                            row[key] = str(detected).lower()
                        else:
                            row[key] = "Error (Invalid Format)"
                else:
                    for key in signals_keys:
                        row[key] = "Parser Failed"
            except Exception as e:
                print(f"  -> Error processing {url}: {e}")
            
            writer.writerow(row)
            outfile.flush() # Write immediately to save progress

    print(f"\n--- BATCH PROCESSING COMPLETE ---")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
