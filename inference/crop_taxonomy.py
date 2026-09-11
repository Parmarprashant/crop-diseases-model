from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class ClassMetadata:
    """
    Formal agricultural metadata for a closed-set disease/pest class.
    Prevents closed-set classifiers from hallucinating cross-crop or cross-part conditions.
    """
    class_id: int
    class_name: str
    crop: str                     # "cotton", "rice", "wheat", "maize", "sugarcane"
    condition_type: str           # "fungal", "bacterial", "viral", "pest", "healthy"
    supported_plant_parts: List[str] = field(default_factory=list) # ["leaf", "stem", "panicle", "boll", "ear"]
    supported_stages: List[str] = field(default_factory=list)      # ["seedling", "vegetative", "flowering", "maturity"]
    common_name: str = ""
    pathogen: str = ""
    training_distribution: str = "MAIN DATA foliar dataset"


# Master taxonomy for all 42 classes in weights/class_names.txt
CLASS_TAXONOMY: Dict[str, ClassMetadata] = {
    "American Bollworm on Cotton": ClassMetadata(
        class_id=0, class_name="American Bollworm on Cotton", crop="cotton",
        condition_type="pest", supported_plant_parts=["boll", "flower", "leaf"],
        common_name="American Bollworm", pathogen="Helicoverpa armigera"
    ),
    "Anthracnose on Cotton": ClassMetadata(
        class_id=1, class_name="Anthracnose on Cotton", crop="cotton",
        condition_type="fungal", supported_plant_parts=["leaf", "boll"],
        common_name="Cotton Anthracnose", pathogen="Colletotrichum gossypii"
    ),
    "Army worm": ClassMetadata(
        class_id=2, class_name="Army worm", crop="maize",
        condition_type="pest", supported_plant_parts=["leaf", "stem"],
        common_name="Armyworm / Fall Armyworm", pathogen="Spodoptera frugiperda"
    ),
    "Becterial Blight in Rice": ClassMetadata(
        class_id=3, class_name="Becterial Blight in Rice", crop="rice",
        condition_type="bacterial", supported_plant_parts=["leaf"],
        common_name="Bacterial Leaf Blight (BLB)", pathogen="Xanthomonas oryzae pv. oryzae"
    ),
    "Brownspot": ClassMetadata(
        class_id=4, class_name="Brownspot", crop="rice",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Rice Brown Spot", pathogen="Bipolaris oryzae"
    ),
    "Common_Rust": ClassMetadata(
        class_id=5, class_name="Common_Rust", crop="maize",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Common Rust of Maize", pathogen="Puccinia sorghi"
    ),
    "Cotton Aphid": ClassMetadata(
        class_id=6, class_name="Cotton Aphid", crop="cotton",
        condition_type="pest", supported_plant_parts=["leaf", "stem"],
        common_name="Cotton Aphid", pathogen="Aphis gossypii"
    ),
    "Flag Smut": ClassMetadata(
        class_id=7, class_name="Flag Smut", crop="wheat",
        condition_type="fungal", supported_plant_parts=["leaf", "stem"],
        common_name="Flag Smut of Wheat", pathogen="Urocystis agropyri"
    ),
    "Gray_Leaf_Spot": ClassMetadata(
        class_id=8, class_name="Gray_Leaf_Spot", crop="maize",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Gray Leaf Spot", pathogen="Cercospora zeae-maydis"
    ),
    "Healthy Maize": ClassMetadata(
        class_id=9, class_name="Healthy Maize", crop="maize",
        condition_type="healthy", supported_plant_parts=["leaf", "stem", "ear"],
        common_name="Healthy Maize Foliage", pathogen="None"
    ),
    "Healthy Wheat": ClassMetadata(
        class_id=10, class_name="Healthy Wheat", crop="wheat",
        condition_type="healthy", supported_plant_parts=["leaf", "stem", "spike"],
        common_name="Healthy Wheat Canopy", pathogen="None"
    ),
    "Healthy cotton": ClassMetadata(
        class_id=11, class_name="Healthy cotton", crop="cotton",
        condition_type="healthy", supported_plant_parts=["leaf", "boll"],
        common_name="Healthy Cotton Foliage", pathogen="None"
    ),
    "Leaf Curl": ClassMetadata(
        class_id=12, class_name="Leaf Curl", crop="cotton",
        condition_type="viral", supported_plant_parts=["leaf"],
        common_name="Cotton Leaf Curl Virus (CLCuD)", pathogen="Begomovirus (Whitefly-transmitted)"
    ),
    "Leaf smut": ClassMetadata(
        class_id=13, class_name="Leaf smut", crop="rice",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Rice Leaf Smut", pathogen="Entyloma oryzae"
    ),
    "Mosaic sugarcane": ClassMetadata(
        class_id=14, class_name="Mosaic sugarcane", crop="sugarcane",
        condition_type="viral", supported_plant_parts=["leaf"],
        common_name="Sugarcane Mosaic Virus", pathogen="Potyvirus"
    ),
    "RedRot sugarcane": ClassMetadata(
        class_id=15, class_name="RedRot sugarcane", crop="sugarcane",
        condition_type="fungal", supported_plant_parts=["stem", "leaf"],
        common_name="Sugarcane Red Rot", pathogen="Colletotrichum falcatum"
    ),
    "RedRust sugarcane": ClassMetadata(
        class_id=16, class_name="RedRust sugarcane", crop="sugarcane",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Sugarcane Orange / Red Rust", pathogen="Puccinia kuehnii"
    ),
    "Rice Blast": ClassMetadata(
        class_id=17, class_name="Rice Blast", crop="rice",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Rice Leaf Blast", pathogen="Magnaporthe oryzae"
    ),
    "Sugarcane Healthy": ClassMetadata(
        class_id=18, class_name="Sugarcane Healthy", crop="sugarcane",
        condition_type="healthy", supported_plant_parts=["leaf", "stem"],
        common_name="Healthy Sugarcane Foliage", pathogen="None"
    ),
    "Tungro": ClassMetadata(
        class_id=19, class_name="Tungro", crop="rice",
        condition_type="viral", supported_plant_parts=["leaf"],
        common_name="Rice Tungro Virus", pathogen="RTBV & RTSV complex"
    ),
    "Wheat Brown leaf Rust": ClassMetadata(
        class_id=20, class_name="Wheat Brown leaf Rust", crop="wheat",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Wheat Brown / Leaf Rust", pathogen="Puccinia triticina"
    ),
    "Wheat Stem fly": ClassMetadata(
        class_id=21, class_name="Wheat Stem fly", crop="wheat",
        condition_type="pest", supported_plant_parts=["stem", "leaf"],
        common_name="Wheat Stem Fly / Shoot Fly", pathogen="Atherigona soccata"
    ),
    "Wheat aphid": ClassMetadata(
        class_id=22, class_name="Wheat aphid", crop="wheat",
        condition_type="pest", supported_plant_parts=["leaf", "spike"],
        common_name="Wheat Aphid", pathogen="Rhopalosiphum padi / Sitobion avenae"
    ),
    "Wheat black rust": ClassMetadata(
        class_id=23, class_name="Wheat black rust", crop="wheat",
        condition_type="fungal", supported_plant_parts=["stem", "leaf"],
        common_name="Wheat Black / Stem Rust", pathogen="Puccinia graminis"
    ),
    "Wheat leaf blight": ClassMetadata(
        class_id=24, class_name="Wheat leaf blight", crop="wheat",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Wheat Leaf Blight / Spot Blotch", pathogen="Bipolaris sorokiniana"
    ),
    "Wheat mite": ClassMetadata(
        class_id=25, class_name="Wheat mite", crop="wheat",
        condition_type="pest", supported_plant_parts=["leaf"],
        common_name="Brown Wheat Mite", pathogen="Petrobia latens"
    ),
    "Wheat powdery mildew": ClassMetadata(
        class_id=26, class_name="Wheat powdery mildew", crop="wheat",
        condition_type="fungal", supported_plant_parts=["leaf", "stem"],
        common_name="Wheat Powdery Mildew", pathogen="Blumeria graminis"
    ),
    "Wheat scab": ClassMetadata(
        class_id=27, class_name="Wheat scab", crop="wheat",
        condition_type="fungal", supported_plant_parts=["spike", "leaf"],
        common_name="Wheat Scab / Fusarium Head Blight", pathogen="Fusarium graminearum"
    ),
    "Wheat___Yellow_Rust": ClassMetadata(
        class_id=28, class_name="Wheat___Yellow_Rust", crop="wheat",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Wheat Stripe / Yellow Rust", pathogen="Puccinia striiformis"
    ),
    "Wilt": ClassMetadata(
        class_id=29, class_name="Wilt", crop="cotton",
        condition_type="fungal", supported_plant_parts=["leaf", "stem"],
        common_name="Cotton Vascular Wilt", pathogen="Fusarium oxysporum / Verticillium dahliae"
    ),
    "Yellow Rust Sugarcane": ClassMetadata(
        class_id=30, class_name="Yellow Rust Sugarcane", crop="sugarcane",
        condition_type="fungal", supported_plant_parts=["leaf"],
        common_name="Sugarcane Brown / Yellow Rust", pathogen="Puccinia melanocephala"
    ),
    "bacterial_blight in Cotton": ClassMetadata(
        class_id=31, class_name="bacterial_blight in Cotton", crop="cotton",
        condition_type="bacterial", supported_plant_parts=["leaf", "stem"],
        common_name="Cotton Bacterial Blight", pathogen="Xanthomonas citri pv. malvacearum"
    ),
    "bollrot on Cotton": ClassMetadata(
        class_id=32, class_name="bollrot on Cotton", crop="cotton",
        condition_type="fungal", supported_plant_parts=["boll"],
        common_name="Cotton Boll Rot Complex", pathogen="Diplodia / Fusarium complex"
    ),
    "bollworm on Cotton": ClassMetadata(
        class_id=33, class_name="bollworm on Cotton", crop="cotton",
        condition_type="pest", supported_plant_parts=["boll", "stem"],
        common_name="Spotted Bollworm", pathogen="Earias vittella"
    ),
    "cotton mealy bug": ClassMetadata(
        class_id=34, class_name="cotton mealy bug", crop="cotton",
        condition_type="pest", supported_plant_parts=["leaf", "stem"],
        common_name="Cotton Mealybug", pathogen="Phenacoccus solenopsis"
    ),
    "cotton whitefly": ClassMetadata(
        class_id=35, class_name="cotton whitefly", crop="cotton",
        condition_type="pest", supported_plant_parts=["leaf"],
        common_name="Cotton Whitefly", pathogen="Bemisia tabaci"
    ),
    "maize ear rot": ClassMetadata(
        class_id=36, class_name="maize ear rot", crop="maize",
        condition_type="fungal", supported_plant_parts=["ear"],
        common_name="Maize Ear Rot", pathogen="Fusarium verticillioides / Aspergillus flavus"
    ),
    "maize fall armyworm": ClassMetadata(
        class_id=37, class_name="maize fall armyworm", crop="maize",
        condition_type="pest", supported_plant_parts=["leaf", "stem", "ear"],
        common_name="Fall Armyworm on Maize", pathogen="Spodoptera frugiperda"
    ),
    "maize stem borer": ClassMetadata(
        class_id=38, class_name="maize stem borer", crop="maize",
        condition_type="pest", supported_plant_parts=["stem", "leaf"],
        common_name="Maize Stem Borer", pathogen="Chilo partellus"
    ),
    "pink bollworm in cotton": ClassMetadata(
        class_id=39, class_name="pink bollworm in cotton", crop="cotton",
        condition_type="pest", supported_plant_parts=["boll", "flower"],
        common_name="Pink Bollworm", pathogen="Pectinophora gossypiella"
    ),
    "red cotton bug": ClassMetadata(
        class_id=40, class_name="red cotton bug", crop="cotton",
        condition_type="pest", supported_plant_parts=["boll", "leaf"],
        common_name="Red Cotton Stainer", pathogen="Dysdercus cingulatus"
    ),
    "thirps on  cotton": ClassMetadata(
        class_id=41, class_name="thirps on  cotton", crop="cotton",
        condition_type="pest", supported_plant_parts=["leaf"],
        common_name="Cotton Thrips", pathogen="Thrips tabaci"
    ),
}


# Precomputed index mappings
SUPPORTED_CROPS = {"cotton", "rice", "wheat", "maize", "sugarcane"}
SUPPORTED_PLANT_PARTS = {"leaf", "stem", "panicle", "ear", "boll", "flower", "root", "unknown"}

# Crop-to-class list mapping
CROP_TO_CLASSES: Dict[str, List[str]] = {}
for name, meta in CLASS_TAXONOMY.items():
    CROP_TO_CLASSES.setdefault(meta.crop, []).append(name)


def get_class_metadata(class_name: Optional[str]) -> Optional[ClassMetadata]:
    """Retrieve formal metadata for a class name."""
    if not class_name:
        return None
    if class_name in CLASS_TAXONOMY:
        return CLASS_TAXONOMY[class_name]
    # Try normalized match
    norm = class_name.lower().replace("_", " ").strip()
    for k, v in CLASS_TAXONOMY.items():
        if k.lower().replace("_", " ").strip() == norm:
            return v
    return None


def check_crop_compatibility(predicted_crop: str, candidate_disease: Optional[str]) -> Tuple[bool, str]:
    """
    STRICT CROP COMPATIBILITY CHECK:
    Returns (True, "") if candidate_disease belongs to predicted_crop.
    Returns (False, reason) if candidate_disease violates crop consistency.
    """
    if not candidate_disease:
        return True, "No candidate disease to evaluate"

    crop_clean = predicted_crop.lower().strip()
    if crop_clean == "unknown":
        return True, "Crop is unverified; cannot enforce cross-crop exclusion"
    
    meta = get_class_metadata(candidate_disease)
    if not meta:
        return False, f"Unknown candidate disease '{candidate_disease}' not in taxonomy"
        
    if meta.crop != crop_clean:
        return False, (
            f"Crop Incompatibility Violation: Detected crop is '{crop_clean}', "
            f"but predicted condition '{candidate_disease}' belongs to '{meta.crop}'."
        )
    return True, ""


def check_plant_part_compatibility(
    crop: str,
    plant_part: str,
    candidate_disease: str
) -> Tuple[bool, str]:
    """
    STRICT PLANT-PART COMPATIBILITY CHECK:
    Verifies that the primary model supports diagnosing the candidate disease
    on the detected plant part.
    """
    crop_clean = crop.lower().strip()
    part_clean = plant_part.lower().strip()

    if part_clean == "unknown":
        return True, "Plant part is unverified; defaulting to defensive check"

    meta = get_class_metadata(candidate_disease)
    if not meta:
        return False, f"Unknown candidate disease '{candidate_disease}'"

    # Specific Rice check: 42-class CNN Rice classes ONLY support leaves
    if crop_clean == "rice" and part_clean in ["panicle", "ear", "grain"]:
        return False, (
            "Plant-Part Incompatibility: Primary 42-class CNN only supports Rice Leaf diseases. "
            f"Detected plant part is '{part_clean}'. Panicle/grain conditions (e.g. False Smut, "
            "Grain Discoloration) are out-of-distribution for the foliar model."
        )

    if part_clean not in meta.supported_plant_parts:
        return False, (
            f"Plant-Part Incompatibility: Condition '{candidate_disease}' affects "
            f"{meta.supported_plant_parts}, but detected plant part is '{part_clean}'."
        )

    return True, ""


# Explicit Whitelisted Legitimate Taxonomy Aliases
# Strictly prevents accidental fuzzy matching from conflating distinct conditions (e.g. blast != blight != smut)
APPROVED_TAXONOMY_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "cotton": {
        "Anthracnose on Cotton": ["anthracnose", "cotton anthracnose", "colletotrichum gossypii", "anthracnose on cotton"],
        "bacterial_blight in Cotton": ["bacterial blight", "angular leaf spot", "bacterial blight in cotton", "cotton bacterial blight", "black arm", "xanthomonas citri pv. malvacearum"],
        "bollrot on Cotton": ["boll rot", "cotton boll rot", "bollrot on cotton", "boll rot complex"],
        "Leaf Curl": ["cotton leaf curl", "leaf curl virus", "clcud", "leaf curl", "cotton leaf curl disease"],
        "Wilt": ["fusarium wilt", "verticillium wilt", "wilt in cotton", "cotton wilt", "vascular wilt", "wilt"],
        "American Bollworm on Cotton": ["american bollworm", "bollworm", "helicoverpa armigera", "american bollworm on cotton"],
        "pink bollworm in cotton": ["pink bollworm", "pectinophora gossypiella", "pink bollworm in cotton"],
        "Cotton Aphid": ["cotton aphid", "aphid", "aphis gossypii"],
        "cotton whitefly": ["cotton whitefly", "whitefly", "bemisia tabaci"],
        "thirps on  cotton": ["thrips", "cotton thrips", "thirps on cotton", "thirps on  cotton", "thrips tabaci"],
        "red cotton bug": ["red cotton bug", "cotton stainer", "dysdercus cingulatus"],
        "Healthy cotton": ["healthy", "healthy cotton", "healthy cotton foliage"]
    },
    "rice": {
        "Rice Blast": ["blast", "rice blast", "magnaporthe oryzae", "leaf blast", "pyricularia oryzae"],
        "Becterial Blight in Rice": ["bacterial blight", "bacterial leaf blight", "blb", "bacterial blight in rice", "becterial blight in rice", "xanthomonas oryzae pv. oryzae", "xanthomonas oryzae"],
        "Brownspot": ["brown spot", "rice brown spot", "brownspot", "bipolaris oryzae"],
        "Leaf smut": ["leaf smut", "rice leaf smut", "entyloma oryzae"],
        "Tungro": ["tungro", "rice tungro", "rice tungro virus"],
        "Healthy Rice": ["healthy", "healthy rice", "healthy paddy"]
    },
    "wheat": {
        "wheat brown leaf rust": ["brown rust", "leaf rust", "wheat leaf rust", "wheat brown leaf rust", "puccinia triticina"],
        "Flag Smut": ["flag smut", "wheat flag smut", "urocystis agropyri"],
        "Healthy Wheat": ["healthy", "healthy wheat", "healthy wheat canopy"],
        "Septoria": ["septoria", "septoria leaf blotch", "septoria tritici", "zymoseptoria tritici"],
        "Powdery Mildew": ["powdery mildew", "wheat powdery mildew", "blumeria graminis"]
    },
    "maize": {
        "Common_Rust": ["common rust", "maize common rust", "corn rust", "puccinia sorghi", "common_rust"],
        "Gray_Leaf_Spot": ["gray leaf spot", "cercospora zeae-maydis", "cercospora zeae maydis", "grey leaf spot", "gray_leaf_spot"],
        "maize ear rot": ["ear rot", "maize ear rot", "fusarium ear rot", "aspergillus ear rot"],
        "maize stem borer": ["stem borer", "maize stem borer", "chilo partellus"],
        "maize fall armyworm": ["fall armyworm", "army worm", "armyworm", "spodoptera frugiperda", "maize fall armyworm", "fall armyworm on maize"],
        "Healthy Maize": ["healthy", "healthy maize", "healthy corn"]
    },
    "sugarcane": {
        "RedRot sugarcane": ["red rot", "sugarcane red rot", "colletotrichum falcatum", "redrot sugarcane"],
        "RedRust sugarcane": ["red rust", "sugarcane rust", "puccinia melanocephala", "redrust sugarcane"],
        "Mosaic sugarcane": ["mosaic", "sugarcane mosaic virus", "mosaic sugarcane"],
        "Healthy Sugarcane": ["healthy", "healthy sugarcane"],
        "Yellow Leaf": ["yellow leaf", "sugarcane yellow leaf virus"]
    }
}


class SemanticDiagnosisMatcher:
    """
    Conservative Semantic Diagnosis Matcher.
    
    CRITICAL SCIENTIFIC SAFETY RULES:
    1. Exact taxonomy match
            ↓
    2. Approved alias match
            ↓
    3. Otherwise: NO MATCH
    
    Never assumes general disease categories (e.g. 'blight', 'spot', 'rot', 'rust')
    are interchangeable across different pathogen classes or crops.
    """

    @staticmethod
    def normalize_string(s: str) -> str:
        if not s:
            return ""
        norm = s.lower().replace("_", " ").replace("-", " ").strip()
        import re
        norm = re.sub(r"\s+", " ", norm)
        return norm

    @classmethod
    def match(
        cls,
        crop1: str,
        plant_part1: str,
        diagnosis1: str,
        crop2: str,
        plant_part2: str,
        diagnosis2: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate if two diagnoses semantically refer to the exact same botanical condition.
        Returns (is_match, explanation, match_telemetry).
        """
        c1 = cls.normalize_string(crop1)
        c2 = cls.normalize_string(crop2)
        p1 = cls.normalize_string(plant_part1)
        p2 = cls.normalize_string(plant_part2)
        d1 = cls.normalize_string(diagnosis1)
        d2 = cls.normalize_string(diagnosis2)

        telemetry = {
            "crop_match": False,
            "plant_part_match": False,
            "disease_match": False,
            "match_level": "NONE",
            "canonical_class": None
        }

        if not d1 or not d2 or d1 in ["unknown", "none"] or d2 in ["unknown", "none"]:
            return False, "One or both diagnoses are unknown/empty", telemetry

        # Crop evaluation: Mismatch between two known, different crops is an immediate rejection
        known_crops = {"cotton", "rice", "wheat", "maize", "sugarcane"}
        crop_match = False
        if c1 in known_crops and c2 in known_crops:
            crop_match = (c1 == c2)
            if not crop_match:
                telemetry["crop_match"] = False
                return False, f"Crop conflict: '{c1}' vs '{c2}'", telemetry
        else:
            # One or both crops are unverified or broad
            crop_match = (c1 == c2 or c1 == "unknown" or c2 == "unknown")
        telemetry["crop_match"] = crop_match

        # Plant part evaluation
        known_parts = {"leaf", "panicle", "stem", "boll", "ear", "root", "flower"}
        part_match = False
        if p1 in known_parts and p2 in known_parts:
            # Panicle vs Leaf is a strict mismatch
            part_match = (p1 == p2)
        else:
            part_match = True
        telemetry["plant_part_match"] = part_match

        # 1. Tier 1: Exact string or exact taxonomy lookup
        meta1 = get_class_metadata(diagnosis1)
        meta2 = get_class_metadata(diagnosis2)
        if meta1 and meta2 and meta1.class_name == meta2.class_name:
            telemetry["disease_match"] = True
            telemetry["match_level"] = "EXACT_TAXONOMY"
            telemetry["canonical_class"] = meta1.class_name
            return True, f"Exact taxonomy match: '{meta1.class_name}'", telemetry

        if d1 == d2:
            telemetry["disease_match"] = True
            telemetry["match_level"] = "EXACT_NAME"
            telemetry["canonical_class"] = meta1.class_name if meta1 else (meta2.class_name if meta2 else d1)
            return True, f"Exact string match: '{d1}'", telemetry

        # 2. Tier 2: Whitelisted Alias Matching within crop
        target_crop = c1 if c1 in known_crops else (c2 if c2 in known_crops else None)
        if target_crop and target_crop in APPROVED_TAXONOMY_ALIASES:
            crop_aliases = APPROVED_TAXONOMY_ALIASES[target_crop]
            for canonical_name, aliases in crop_aliases.items():
                norm_canon = cls.normalize_string(canonical_name)
                norm_aliases = [cls.normalize_string(a) for a in aliases] + [norm_canon]
                
                d1_matches = (d1 in norm_aliases or d1 == norm_canon)
                d2_matches = (d2 in norm_aliases or d2 == norm_canon)
                
                if d1_matches and d2_matches:
                    telemetry["disease_match"] = True
                    telemetry["match_level"] = "WHITELISTED_ALIAS"
                    telemetry["canonical_class"] = canonical_name
                    return True, f"Whitelisted alias match: '{d1}' and '{d2}' map to canonical '{canonical_name}'", telemetry

        # 3. Tier 3: Strict Otherwise -> NO MATCH
        telemetry["disease_match"] = False
        telemetry["match_level"] = "NO_MATCH"
        return False, f"No approved semantic match between '{diagnosis1}' and '{diagnosis2}'", telemetry

