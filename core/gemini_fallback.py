import os
import io
import json
import base64
from typing import Dict, Any, Optional, List
from PIL import Image

try:
    from google import genai
    from google.genai import types
    NEW_GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    NEW_GOOGLE_GENAI_AVAILABLE = False

try:
    import google.generativeai as legacy_genai
    LEGACY_GENAI_AVAILABLE = True
except ImportError:
    LEGACY_GENAI_AVAILABLE = False


class GeminiVisionFallback:
    """
    Smart Fallback engine powered by Gemini 2.0 Flash / 1.5 Pro Vision.
    Invoked exclusively when the primary EfficientNet-B5 + CBAM model confidence
    falls below the threshold (0.80).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self.client = None
        self._init_client()

    def refresh_key(self) -> bool:
        """Dynamically reload GEMINI_API_KEY and GEMINI_MODEL from .env or environment."""
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except Exception:
            pass

        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        env_model = os.environ.get("GEMINI_MODEL", self.model_name).strip()
        if env_key and env_key != "your_gemini_api_key_here":
            if env_key != self.api_key or self.client is None:
                self.api_key = env_key
                self.model_name = env_model
                self._init_client()
                print(f"[GeminiFallback] Reloaded active GEMINI_API_KEY (prefix: {self.api_key[:6]}...) with model: {self.model_name}")
                return True
            return True
        return False

    def _init_client(self):
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return

        if NEW_GOOGLE_GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiFallback] Notice initializing google.genai: {e}")
        elif LEGACY_GENAI_AVAILABLE:
            try:
                legacy_genai.configure(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiFallback] Notice initializing legacy google.generativeai: {e}")

    def call_fallback(
        self,
        image: Image.Image,
        preliminary_predictions: Optional[List[Dict[str, Any]]] = None,
        primary_confidence: Optional[float] = None,
        severity_pct: Optional[float] = None,
        pests_detected: Optional[List[str]] = None,
        user_crop_hint: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an INDEPENDENT visual assessment of the original image using Gemini Vision.
        CRITICAL SCIENTIFIC RULE:
        Zero exposure to CNN logits, candidate predictions, or preliminary confidences.
        Gemini receives only the original image, diagnostic instructions, and structured schema.
        """
        # Ensure latest API key is refreshed from .env
        self.refresh_key()

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return {
                "status": "UNAVAILABLE",
                "crop": "unknown",
                "plant_part": "unknown",
                "diagnosis": "Unavailable",
                "assessment_strength": "LOW",
                "model_reported_confidence": 0.0,
                "reasoning_summary": "Gemini Vision is unavailable because GEMINI_API_KEY is not configured.",
                "evidence": []
            }

        prompt = (
            "You are an expert plant pathologist and agronomist examining an agricultural field photograph.\n"
            "Provide an independent, unbiased diagnostic evaluation of this plant foliage/tissue.\n\n"
            "CRITICAL SCIENTIFIC INSTRUCTIONS:\n"
            "1. Identify the crop species (e.g. Cotton, Rice, Wheat, Maize, Sugarcane, Mango, Tomato, etc., or 'unknown').\n"
            "2. Identify the visible plant organ (leaf, panicle/earhead, stem, boll, fruit, flower, root, or 'unknown').\n"
            "3. Describe the visible symptoms (color, shape, lesion margins, pustules, fungal structures).\n"
            "4. Do NOT claim absolute certainty or declare an 'exact confirmed pathogen' based solely on an image.\n"
            "5. If visual evidence is ambiguous, degraded, out-of-distribution, or insufficient, set status to 'UNCERTAIN', 'UNKNOWN', or 'INSUFFICIENT_EVIDENCE', and diagnosis to 'Unknown Condition'.\n"
            "6. For assessment_strength, assign 'HIGH', 'MEDIUM', or 'LOW' based on symptom clarity and sharpness.\n"
            "7. Output concise evidence summary and diagnostic reasoning without chain-of-thought.\n\n"
            "Output a valid JSON object strictly matching this schema:\n"
            "{\n"
            '  "status": "SUCCESS",\n'
            '  "crop": "crop name or unknown",\n'
            '  "plant_part": "leaf | panicle | stem | boll | fruit | flower | root | unknown",\n'
            '  "diagnosis": "Condition name, or Unknown Condition",\n'
            '  "assessment_strength": "HIGH | MEDIUM | LOW",\n'
            '  "model_reported_confidence": 0.85,\n'
            '  "reasoning_summary": "Concise visual evidence and symptoms",\n'
            '  "evidence": ["visual symptom 1", "visual symptom 2"]\n'
            "}\n"
            "Do not include markdown code block formatting outside the JSON."
        )

        # 1. Try google-genai SDK with resilient model list
        if self.client and NEW_GOOGLE_GENAI_AVAILABLE:
            candidate_models = [self.model_name, "gemini-2.5-flash", "gemini-2.5-pro"]
            seen_models = []
            for m in candidate_models:
                if m and m not in seen_models:
                    seen_models.append(m)

            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            for m_id in seen_models:
                try:
                    response = self.client.models.generate_content(
                        model=m_id,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            prompt
                        ]
                    )
                    if response and response.text:
                        return self._parse_gemini_response(response.text)
                except Exception as e:
                    print(f"[GeminiFallback] API call with model '{m_id}' failed: {e}")

        # 2. Try legacy google-generativeai SDK
        if LEGACY_GENAI_AVAILABLE and self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                model = legacy_genai.GenerativeModel(self.model_name)
                response = model.generate_content([image, prompt])
                if response and response.text:
                    return self._parse_gemini_response(response.text)
            except Exception as e:
                print(f"[GeminiFallback] Legacy API call error: {e}")

        # 3. Fail closed if offline, key invalid, or API unreachable - NEVER fake a fallback
        return {
            "status": "ERROR",
            "crop": "unknown",
            "plant_part": "unknown",
            "diagnosis": "Error",
            "assessment_strength": "LOW",
            "model_reported_confidence": 0.0,
            "reasoning_summary": "Gemini Vision API call failed, timed out, or is unreachable.",
            "evidence": []
        }

    def _parse_gemini_response(self, text: str) -> Dict[str, Any]:
        """Parse structured JSON from Gemini's response string with defensive boundary enforcement."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            diag = (data.get("diagnosis") or data.get("disease_name") or "Unknown Condition").strip()
            status = data.get("status", "SUCCESS").upper()

            if diag.upper() in ["UNKNOWN", "INSUFFICIENT_EVIDENCE", "UNRESOLVED", "NONE"]:
                diag = "Unknown Condition"
                if status == "SUCCESS":
                    status = "UNKNOWN"

            # Parse assessment_strength and model_reported_confidence
            strength = str(data.get("assessment_strength", "MEDIUM")).upper()
            if strength not in ["HIGH", "MEDIUM", "LOW"]:
                strength = "MEDIUM"

            raw_conf = data.get("model_reported_confidence", data.get("confidence", 0.5))
            try:
                raw_conf = float(raw_conf)
            except (ValueError, TypeError):
                raw_conf = 0.5

            reasoning = data.get("reasoning_summary") or data.get("visual_reasoning") or "Visual examination completed."
            evidence = data.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]

            return {
                "status": status,
                "crop": (data.get("crop") or "unknown").lower().strip(),
                "plant_part": (data.get("plant_part") or "unknown").lower().strip(),
                "diagnosis": diag,
                "disease_name": diag,  # legacy compatibility
                "assessment_strength": strength,
                "model_reported_confidence": raw_conf,
                "confidence": raw_conf, # legacy compatibility
                "reasoning_summary": reasoning,
                "visual_reasoning": reasoning, # legacy compatibility
                "evidence": evidence
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "crop": "unknown",
                "plant_part": "unknown",
                "diagnosis": "Unknown Condition",
                "disease_name": "Unknown Condition",
                "assessment_strength": "LOW",
                "model_reported_confidence": 0.0,
                "confidence": 0.0,
                "reasoning_summary": f"Failed to parse structured response from vision model: {e}",
                "visual_reasoning": "Failed to parse structured response from vision model.",
                "evidence": []
            }

