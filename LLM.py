import json
import re
from typing import Any, Dict, List, Optional

import boto3

DEFAULT_ENDPOINT_NAME = "jumpstart-dft-llama-3-1-8b-instruct-20260416-102920"


class CrawlPreprocessor:
    """
    Light preprocessing only.
    No final decision logic here.
    The LLM makes all detection decisions.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def _safe_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def _lower_list(self, items: List[Any]) -> List[str]:
        return [self._safe_text(x).strip() for x in (items or []) if self._safe_text(x).strip()]

    def _extract_prices(self, text: str) -> List[float]:
        prices = []
        patterns = [
            r"\$ ?([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?)",
            r"\$ ?([0-9]+(?:\.[0-9]{2})?)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                try:
                    prices.append(float(match.replace(",", "")))
                except ValueError:
                    pass
        return sorted(set(prices))

    def _extract_candidate_locations(self, nav_text: List[str], visible_text: str) -> List[str]:
        known_patterns = [
            "round rock",
            "broadway square",
            "tyler",
            "dallas",
            "mississauga",
            "canada",
        ]
        combined = " ".join(nav_text).lower() + " " + visible_text.lower()
        found = [x for x in known_patterns if x in combined]
        return sorted(set(found))

    def build_llm_input(self) -> Dict[str, Any]:
        visible_text = self._safe_text(self.data.get("visible_text"))
        nav_text = self._lower_list(self.data.get("nav_text", []))
        internal_links = self._lower_list(self.data.get("internal_links", []))
        external_links = self._lower_list(self.data.get("external_links", []))
        script_srcs = self._lower_list(self.data.get("script_srcs", []))
        tech_hints = self._lower_list(self.data.get("tech_hints", []))

        prices = self._extract_prices(visible_text)
        candidate_locations = self._extract_candidate_locations(nav_text, visible_text)

        distilled = {
            "url": self.data.get("url"),
            "final_url": self.data.get("final_url"),
            "title": self.data.get("title"),
            "meta_description": self.data.get("meta_description"),
            "visible_text_excerpt": visible_text[:5000],  # Truncate more aggressively
            "nav_text": nav_text[:50],
            "internal_links": internal_links[:50],
            "external_links": external_links[:50],
            "script_srcs": script_srcs[:50],
            "tech_hints": tech_hints,
            "derived_features": {
                "prices_found": prices[:100],
                "has_price_over_1000": any(p >= 1000 for p in prices),
                "has_add_to_cart_phrase": "add to cart" in visible_text.lower(),
                "has_buy_now_phrase": "buy now" in visible_text.lower(),
                "has_checkout_phrase": ("checkout" in visible_text.lower()) or ("check out" in visible_text.lower()),
                "product_link_count": sum(1 for x in internal_links if "/products/" in x),
                "collection_link_count": sum(1 for x in internal_links if "/collections/" in x),
                "candidate_locations": candidate_locations,
                "location_link_candidates": [
                    x for x in internal_links
                    if any(k in x for k in ["/store-locator", "/locations", "/find-us", "/our-stores"])
                ][:50],
                "engagement_link_candidates": [
                    x for x in internal_links
                    if any(k in x for k in [
                        "/engagement",
                        "engagement-rings",
                        "/bridal",
                        "/custom-ring",
                        "/custom-ring-builder",
                        "/custom-ring-design",
                        "lab-grown-engagement-rings",
                    ])
                ][:50],
                "custom_ring_link_candidates": [
                    x for x in internal_links
                    if any(k in x for k in [
                        "/custom-ring",
                        "/custom-ring-builder",
                        "/custom-ring-design",
                        "design-your-ring",
                    ])
                ][:50],
                "has_meta_pixel_script": any(
                    ("facebook.net/en_us/fbevents.js" in s.lower()) or ("signals/config/" in s.lower())
                    for s in script_srcs
                ),
                "has_google_ads_script": any(
                    ("gtag/js?id=aw-" in s.lower()) or ("gtm.js?id=" in s.lower()) or ("gtag/js?id=gt-" in s.lower())
                    for s in script_srcs
                ),
                "shopify_like": any(
                    x in " ".join(script_srcs + internal_links).lower()
                    for x in ["myshopify.com", "/cdn/shop/", "shopifycloud", "shop.app", "shopify_pay", "trekkie.storefront"]
                ) or ("shopify" in " ".join(tech_hints).lower()),
            },
        }

        return distilled


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_llama_prompt(prepared: Dict[str, Any]) -> str:
    """
    LLM is the decision-maker.
    It must decide detection for each signal and provide one-line evidence.
    """

    schema = {
        "url": "string",
        "signals": {
            "revenue_brand_signal": {
                "detected": "boolean",
                "evidence": "string"
            },
            "engagement_ring_focus": {
                "detected": "boolean",
                "evidence": "string"
            },
            "ecommerce_maturity": {
                "detected": "boolean",
                "evidence": "string"
            },
            "ads_running": {
                "detected": "boolean",
                "evidence": "string"
            },
            "multi_location_scale": {
                "detected": "boolean",
                "evidence": "string"
            },
            "custom_ring_offering": {
                "detected": "boolean",
                "evidence": "string"
            }
        }
    }

    instructions = f"""
You are a strict website signal detector.

Your job:
Given crawl data from a jewelry retailer website, decide whether each of these signals is present.

Signals:
1. revenue_brand_signal
   Detect true if the site clearly appears premium, luxury, fine jewelry, or carries strong high-value brand/product cues.
   Examples of strong evidence:
   - phrases like "fine jewelry", "luxury", "luxury timepieces"
   - multiple products priced above $1000
   - premium watch/jewelry brands
   Detect false if the signal is weak, unclear, low-end, or not supported.

2. engagement_ring_focus
   Detect true if the site clearly emphasizes engagement rings, bridal, wedding bands, ring builder, or dedicated engagement collections.

3. ecommerce_maturity
   Detect true if the site clearly has mature ecommerce behavior such as add-to-cart, checkout, buy now, products, collections, Shopify/WooCommerce-like structure, strong catalog UX.

4. ads_running
   Detect true if there is evidence of active ad/tracking setup such as Meta Pixel, Facebook events, Google Ads tags, GTM, or Google Ads conversion scripts.
   This is detection of ad-tech presence, not proof of current spend.

5. multi_location_scale
   Detect true if there is evidence of multiple store locations, multiple cities/stores explicitly listed, also mention in the evidence line.

6. custom_ring_offering
   Detect true if there is clear evidence of custom ring design, ring builder, design your ring, or bespoke/custom ring offering.

Rules:
- You must decide TRUE or FALSE for every signal.
- You must provide exactly one concise evidence line per signal.
- Evidence must be grounded only in the provided data.
- Do not score anything.
- Do not add extra commentary.
- Return JSON only.
- No markdown.
- The evidence line should be short and specific.

Return JSON exactly in this structure:
{json.dumps(schema, indent=2)}

Website data:
{json.dumps(prepared, indent=2)}
""".strip()

    return instructions


def invoke_sagemaker_llama(
    prompt: str,
    endpoint_name: str = DEFAULT_ENDPOINT_NAME,
    region_name: Optional[str] = None,
    max_new_tokens: int = 1200,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    runtime = boto3.client("sagemaker-runtime", region_name=region_name)

    # Llama 3 prompt format as used in other files in this project
    payload = {
        "inputs": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a strict website signal detector. Respond ONLY with valid JSON."
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt.strip()}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 0.1,
            "top_p": top_p,
            "do_sample": True if temperature > 0 else False,
            "return_full_text": False,
        },
    }

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    body = response["Body"].read().decode("utf-8")
    parsed = json.loads(body)
    return parsed


def try_extract_text_generation(raw_response: Dict[str, Any]) -> str:
    """
    Handles a few common SageMaker / JumpStart response shapes.
    Adjust if your endpoint returns a different shape.
    """
    if isinstance(raw_response, dict):
        if "choices" in raw_response and raw_response["choices"]:
            choice = raw_response["choices"][0]
            if isinstance(choice, dict):
                if "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content", "")
                if "text" in choice:
                    return choice["text"]

        if "generation" in raw_response:
            return raw_response["generation"]

        if "generated_text" in raw_response:
            return raw_response["generated_text"]

        if "outputs" in raw_response and raw_response["outputs"]:
            first = raw_response["outputs"][0]
            if isinstance(first, dict):
                return first.get("text", "") or first.get("generated_text", "")
            if isinstance(first, str):
                return first

    if isinstance(raw_response, list) and raw_response:
        first = raw_response[0]
        if isinstance(first, dict):
            return first.get("generated_text", "") or first.get("text", "")
        if isinstance(first, str):
            return first

    raise ValueError(f"Could not extract generated text from endpoint response: {raw_response}")


def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Attempts to parse strict JSON.
    If model adds extra text accidentally, extracts the first JSON object block.
    """
    text = text.strip()

    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("LLM output is not valid JSON.")


def detect_signals_with_llm(
    input_path: str,
    endpoint_name: str = DEFAULT_ENDPOINT_NAME,
    region_name: Optional[str] = None,
    prepared_output_path: Optional[str] = "prepared_input.json",
    raw_llm_output_path: Optional[str] = "raw_llm_response.json",
    final_output_path: Optional[str] = "signal_detection_output.json",
) -> Dict[str, Any]:
    raw_data = load_json(input_path)

    preprocessor = CrawlPreprocessor(raw_data)
    prepared = preprocessor.build_llm_input()

    if prepared_output_path:
        save_json(prepared_output_path, prepared)

    prompt = build_llama_prompt(prepared)

    raw_llm_response = invoke_sagemaker_llama(
        prompt=prompt,
        endpoint_name=endpoint_name,
        region_name=region_name,
        max_new_tokens=1200,
        temperature=0.0,
        top_p=0.9,
    )

    if raw_llm_output_path:
        save_json(raw_llm_output_path, raw_llm_response)

    generated_text = try_extract_text_generation(raw_llm_response)
    final_detection = parse_llm_json(generated_text)

    if final_output_path:
        save_json(final_output_path, final_detection)

    return final_detection


if __name__ == "__main__":
    result = detect_signals_with_llm(
        input_path="crawlee_result.json",
        endpoint_name=DEFAULT_ENDPOINT_NAME,
        region_name=None,  # e.g. "us-east-1"
        prepared_output_path="prepared_input.json",
        raw_llm_output_path="raw_llm_response.json",
        final_output_path="signal_detection_output.json",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))