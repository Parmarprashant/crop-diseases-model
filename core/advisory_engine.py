from typing import Dict, Any, List, Optional
import re

# Comprehensive Agricultural Knowledge Base for all 42 MAIN DATA Classes
DISEASE_ADVISORY_DB: Dict[str, Dict[str, Any]] = {
    # ==================== COTTON ====================
    "Healthy cotton": {
        "crop": "Cotton",
        "clean_name": "Healthy Cotton Foliage",
        "category": "Healthy Crop",
        "description": "Foliage shows vibrant green canopy with vigorous vegetative development. No active pest or fungal lesions detected.",
        "chemical_control": ["No chemical intervention required."],
        "organic_control": ["Apply prophylactic liquid seaweed extract or Panchagavya (3%) at 15-day intervals for immunity boost."],
        "precautions": [
            "Maintain optimal drip irrigation; avoid excessive waterlogging in the root zone.",
            "Scout field twice weekly, especially on the underside of top leaves for early sap-sucking pests.",
            "Ensure balanced N-P-K nutrition (avoid excessive Nitrogen which makes leaves succulent to pests)."
        ]
    },
    "bacterial_blight in Cotton": {
        "crop": "Cotton",
        "clean_name": "Bacterial Blight (Angular Leaf Spot)",
        "category": "Bacterial Disease",
        "description": "Caused by Xanthomonas citri pv. malvacearum. Causes angular, water-soaked leaf spots bounded by veinlets, progressing to 'black arm' on stems and boll rot.",
        "chemical_control": [
            "Spray Copper Oxychloride 50 WP (2.5 g/L) + Streptocycline (100 mg/L) at first appearance of angular spots.",
            "Follow up with Copper Hydroxide (2.0 g/L) after 10-12 days if humid weather persists."
        ],
        "organic_control": [
            "Foliar spray of Pseudomonas fluorescens @ 10 g/L or 5 ml/L liquid formulation.",
            "5% Neem Seed Kernel Extract (NSKE) spray to suppress secondary bacterial spread."
        ],
        "precautions": [
            "Delint seed with commercial sulfuric acid before future sowings to eradicate seed-borne inoculum.",
            "Avoid overhead sprinkler irrigation which splashes bacteria between leaves.",
            "Destroy and burn infected crop residues and stubbles immediately after final picking."
        ]
    },
    "Anthracnose on Cotton": {
        "crop": "Cotton",
        "clean_name": "Anthracnose (Colletotrichum)",
        "category": "Fungal Disease",
        "description": "Caused by Colletotrichum gossypii. Produces small, reddish-brown water-soaked spots on cotyledons and leaves, causing seedling damping-off and sunken boll lesions.",
        "chemical_control": [
            "Spray Carbendazim 50% WP @ 1 g/L or Mancozeb 75% WP @ 2.5 g/L.",
            "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L for systemic suppression."
        ],
        "organic_control": [
            "Seed bio-priming with Trichoderma viride @ 10 g/kg seed.",
            "Foliar spray of fermented cow urine extract (5%) mixed with neem oil."
        ],
        "precautions": [
            "Use certified acid-delinted disease-free seed stocks.",
            "Ensure wider plant spacing (90x60 cm) to allow adequate sunlight penetration and air circulation.",
            "Avoid working in the cotton field when foliage is wet with dew or rain."
        ]
    },
    "bollrot on Cotton": {
        "crop": "Cotton",
        "clean_name": "Cotton Boll Rot Complex",
        "category": "Fungal & Bacterial Complex",
        "description": "Fungal-bacterial decay of developing and mature bolls (Diplodia, Fusarium, Xanthomonas), causing dark sunken discoloration and lint staining.",
        "chemical_control": [
            "Spray Propiconazole 25% EC @ 1 ml/L or Copper Oxychloride @ 2.5 g/L directly targeting the fruiting zones.",
            "Tank-mix with Streptocycline (100 ppm) if bacterial water-soaking is observed on young bolls."
        ],
        "organic_control": [
            "Foliar spray of Trichoderma harzianum culture broth @ 5 ml/L.",
            "Neem-garlic extract spray (3%) to repel wound-causing boll-puncturing bugs."
        ],
        "precautions": [
            "Prune lower redundant vegetative branches (chattis) to improve airflow in dense canopies.",
            "Prevent insect puncture wounds by strictly controlling bollworms and red cotton bugs.",
            "Harvest open bolls promptly; do not let mature bolls stay in high humidity."
        ]
    },
    "Leaf Curl": {
        "crop": "Cotton",
        "clean_name": "Cotton Leaf Curl Virus (CLCuD)",
        "category": "Viral Disease",
        "description": "Begomovirus transmitted exclusively by the whitefly (Bemisia tabaci). Symptoms include upward or downward leaf curling, vein thickening, and enation (leaf-like outgrowths).",
        "chemical_control": [
            "Control whitefly vector immediately: Spray Diafenthiuron 50% WP @ 1.2 g/L or Pyriproxyfen 10% EC @ 2 ml/L.",
            "Alternative vector control: Flonicamid 50% WG @ 0.3 g/L or Afidopyropen 50 g/L @ 2 ml/L."
        ],
        "organic_control": [
            "Install yellow sticky traps @ 15-20 traps per acre at canopy height to capture adult whiteflies.",
            "Spray 5% NSKE (Neem Seed Kernel Extract) or Verticillium lecanii @ 5 g/L."
        ],
        "precautions": [
            "Uproot and bury severely infected enation-showing plants during early vegetative stages (first 45 days).",
            "Keep field borders completely free from weed hosts such as Abutilon indicum and Parthenium.",
            "Grow CLCuD-resistant or tolerant Bt cotton hybrids recommended for your zone."
        ]
    },
    "Wilt": {
        "crop": "Cotton",
        "clean_name": "Vascular Fusarium / Verticillium Wilt",
        "category": "Soil-Borne Fungal Disease",
        "description": "Vascular blockage causing yellowing between leaf veins, sudden foliage drooping, leaf drop, and distinctive dark brown discoloration of xylem vessels.",
        "chemical_control": [
            "Spot drench soil around infected root zones with Carbendazim 50 WP @ 2 g/L or Copper Oxychloride @ 3 g/L.",
            "Apply Potassium Nitrate (1%) foliar spray to alleviate vascular water stress."
        ],
        "organic_control": [
            "Heavy application of Trichoderma viride enriched Farm Yard Manure (2.5 kg Trichoderma in 250 kg FYM per acre).",
            "Soil drench with Pseudomonas fluorescens @ 10 ml/L around affected patches."
        ],
        "precautions": [
            "Avoid deep inter-cultivation that causes root injuries and facilitates pathogen entry.",
            "Practice minimum 3-year crop rotation with non-host crops such as sorghum, pearl millet, or paddy.",
            "Apply adequate potash (MOP) to enhance vascular wall thickness."
        ]
    },
    "American Bollworm on Cotton": {
        "crop": "Cotton",
        "clean_name": "American Bollworm (Helicoverpa armigera)",
        "category": "Lepidopteran Insect Pest",
        "description": "Voracious caterpillar feeding on cotton squares, flower buds, and bolls with its body half inside the boll hole and fecal pellets outside.",
        "chemical_control": [
            "Chlorantraniliprole 18.5% SC (Coragen) @ 0.3 ml/L or Emamectin Benzoate 5% SG @ 0.4 g/L.",
            "Flubendiamide 39.35% SC @ 0.25 ml/L or Spinetoram 11.7% SC @ 1 ml/L for severe infestations."
        ],
        "organic_control": [
            "Install Helicoverpa pheromone traps (Helilure) @ 5 traps/acre for pest monitoring.",
            "Spray HaNPV (Helicoverpa Nuclear Polyhedrosis Virus) @ 250 LE/acre in late afternoon.",
            "Release egg parasitoid Trichogramma chilonis @ 60,000/acre at weekly intervals."
        ],
        "precautions": [
            "Plant marigold (Tagetes erecta) or pigeonpea as border/trap crops (1 row for every 10 cotton rows).",
            "Handpick and destroy early instar larvae during square initiation.",
            "Deep summer ploughing to expose pupae to predatory birds and harsh sun."
        ]
    },
    "bollworm on Cotton": {
        "crop": "Cotton",
        "clean_name": "Spotted / Spiny Bollworm (Earias spp.)",
        "category": "Lepidopteran Insect Pest",
        "description": "Earias vittella bore into tender terminal shoots causing terminal wilting, and later bore into bolls causing premature shedding and ruined lint.",
        "chemical_control": [
            "Emamectin Benzoate 5% SG @ 0.4 g/L or Indoxacarb 14.5% SC @ 1 ml/L.",
            "Chlorantraniliprole 18.5% SC @ 0.3 ml/L."
        ],
        "organic_control": [
            "Spray Bacillus thuringiensis (Bt) kurstaki formulation @ 2 g/L.",
            "Install light traps @ 1 per acre to monitor and trap adult moths."
        ],
        "precautions": [
            "Clip and destroy drooping terminal shoots during vegetative growth to eliminate early generation caterpillars.",
            "Avoid ratoon cotton cultivation which sustains pest populations across seasons.",
            "Destroy malvaceous weed hosts like Abutilon and Hibiscus near field boundaries."
        ]
    },
    "pink bollworm in cotton": {
        "crop": "Cotton",
        "clean_name": "Pink Bollworm (Pectinophora gossypiella)",
        "category": "Destructive Internal Boll Pest",
        "description": "Larvae bore into flower buds producing rosette flowers, then enter green bolls where entry holes seal up, chewing seeds and staining lint pink.",
        "chemical_control": [
            "At 8-10 moths/trap/day ETL: Spray Profenofos 50% EC @ 2 ml/L or Chlorpyrifos 20% EC @ 2.5 ml/L.",
            "Spinetoram 11.7% SC @ 1 ml/L or Indoxacarb 14.5% SC @ 1 ml/L for late-stage boll protection."
        ],
        "organic_control": [
            "Install Pheromone traps with Gossyplure @ 8-10 traps/acre.",
            "Deploy PB-Rope L (pheromone mating disruption ropes) @ 100-150 dispensers/acre early in the season."
        ],
        "precautions": [
            "Terminate cotton crop by December/January; strictly avoid extending crop into summer.",
            "Promptly dispose of or burn cotton stalks; do not stack stalks in the field or near ginning yards.",
            "Collect and destroy rosette flowers manually every morning."
        ]
    },
    "Cotton Aphid": {
        "crop": "Cotton",
        "clean_name": "Cotton Aphid (Aphis gossypii)",
        "category": "Sap-Sucking Insect Pest",
        "description": "Colonies of tiny yellowish-green soft-bodied insects suck cell sap from tender shoots and leaf undersides, secreting honeydew that fosters black sooty mold.",
        "chemical_control": [
            "Imidacloprid 17.8% SL @ 0.3 ml/L or Acetamiprid 20% SP @ 0.2 g/L.",
            "Thiamethoxam 25% WG @ 0.25 g/L or Flonicamid 50% WG @ 0.3 g/L."
        ],
        "organic_control": [
            "5% Neem Seed Kernel Extract (NSKE) or commercial Azadirachtin 10,000 ppm @ 1.5 ml/L.",
            "Conserve and encourage natural biological predators: Ladybird beetles (Coccinella) and lacewings."
        ],
        "precautions": [
            "Install yellow sticky cards @ 10-15 per acre.",
            "Avoid excessive and unbalanced urea application which triggers rapid succulent aphid population surges.",
            "Wash foliage with pressurized water jets during early isolated outbreaks."
        ]
    },
    "cotton mealy bug": {
        "crop": "Cotton",
        "clean_name": "Cotton Mealybug (Phenacoccus solenopsis)",
        "category": "Sap-Sucking Invasive Pest",
        "description": "White waxy cottony masses clustering on tender growing tips, stems, and leaf petioles, causing severe stunted growth, boll distortion, and drying.",
        "chemical_control": [
            "Profennofos 50% EC @ 2.5 ml/L or Buprofezin 25% SC @ 2 ml/L mixed with soap/surfactant @ 1 ml/L.",
            "Thiodicarb 75% WP @ 2 g/L for heavy infestations."
        ],
        "organic_control": [
            "Release exotic encyrtid parasitoid wasp Aenasius bambawalei (hugely effective natural bio-control).",
            "Spray Verticillium lecanii or Beauveria bassiana @ 5 g/L with wetting agent."
        ],
        "precautions": [
            "Apply sticky barrier bands (grease/castor oil) on lower stems to prevent symbiotic ants from transporting mealybugs.",
            "Eradicate Parthenium hysterophorus (congress grass) around fields, which is the primary alternate host.",
            "Prune and burn heavily infested individual twigs in polythene bags to prevent waxy crawlers from spreading."
        ]
    },
    "cotton whitefly": {
        "crop": "Cotton",
        "clean_name": "Cotton Whitefly (Bemisia tabaci)",
        "category": "Sap-Sucking Pest & Vector",
        "description": "Tiny powdery-white winged insects congregating on leaf undersides. Sucks sap, causes chlorotic leaf spotting, secretes sooty honeydew, and transmits Leaf Curl Virus.",
        "chemical_control": [
            "Diafenthiuron 50% WP @ 1.2 g/L or Pyriproxyfen 10% EC @ 2 ml/L.",
            "Afidopyropen 50 g/L @ 2 ml/L or Spiromesifen 22.9% SC @ 1 ml/L."
        ],
        "organic_control": [
            "Yellow sticky traps @ 15-20 traps/acre placed 30 cm above crop canopy.",
            "Castor oil smeared yellow plastic sheets placed along field borders.",
            "Spray 5% NSKE or Beauveria bassiana @ 5 g/L."
        ],
        "precautions": [
            "Avoid synthetic pyrethroid sprays early in the season, which destroy natural whitefly predators.",
            "Maintain clean field borders free from malvaceous and solanaceous weeds.",
            "Ensure uniform irrigation; avoid water stress during hot, dry spells."
        ]
    },
    "red cotton bug": {
        "crop": "Cotton",
        "clean_name": "Red Cotton Stainer (Dysdercus cingulatus)",
        "category": "Sucking Bug Pest",
        "description": "Bright red bugs with black marks and white bands. They puncture developing green bolls, feeding on seeds and introducing the fungus Nematospora, staining cotton lint indelible yellow-brown.",
        "chemical_control": [
            "Chlorpyrifos 20% EC @ 2 ml/L or Profenofos 50% EC @ 2 ml/L directly on open bolls.",
            "Deltamethrin 2.8% EC @ 1 ml/L."
        ],
        "organic_control": [
            "Dust field margins with diatomaceous earth or fine wood ash.",
            "Handpick gregarious bug clusters with kerosene-water buckets in smallholder plots."
        ],
        "precautions": [
            "Harvest open bolls immediately without delay to prevent bug feeding.",
            "Remove and destroy alternative weed hosts like wild hollyhock and Abutilon.",
            "Deep tillage post-harvest to destroy eggs laid in soil cracks."
        ]
    },
    "thirps on  cotton": {
        "crop": "Cotton",
        "clean_name": "Cotton Thrips (Thrips tabaci)",
        "category": "Lacerating-Sucking Pest",
        "description": "Slender minute insects scraping leaf epidermis and sucking exuded sap. Causes silvery sheen on leaf undersides, upward cupping of young leaves, and 'bronzing' of older foliage.",
        "chemical_control": [
            "Fipronil 5% SC @ 1.5 ml/L or Imidacloprid 17.8% SL @ 0.3 ml/L.",
            "Spinetoram 11.7% SC @ 1 ml/L or Tolfenpyrad 15% EC @ 2 ml/L for resistant thrips."
        ],
        "organic_control": [
            "Blue sticky traps @ 15-20 per acre (thrips are specifically attracted to blue color).",
            "Spray 5% Neem Seed Kernel Extract (NSKE) + mild soap."
        ],
        "precautions": [
            "Maintain soil moisture through timely irrigation (dry spells accelerate thrips multiplication).",
            "Seed treatment with Imidacloprid 70% WS @ 5 g/kg seed provides 30-day early vegetative protection.",
            "Avoid excess nitrogenous top-dressing."
        ]
    },

    # ==================== WHEAT ====================
    "Healthy Wheat": {
        "crop": "Wheat",
        "clean_name": "Healthy Wheat Canopy",
        "category": "Healthy Crop",
        "description": "Erect, clean green leaves with well-developed tillers. No rust pustules, blights, or insect feeding scars.",
        "chemical_control": ["No chemical fungicide or pesticide required."],
        "organic_control": ["Prophylactic foliar spray of vermiwash (5%) or Jeevamrutha at boot stage to enhance grain filling."],
        "precautions": [
            "Schedule timely irrigations at critical growth stages (Crown Root Initiation, Booting, Flowering, Milk stage).",
            "Scout canopy weekly for early yellow or brown rust foci, especially in cool, humid mornings."
        ]
    },
    "Wheat Brown leaf Rust": {
        "crop": "Wheat",
        "clean_name": "Brown / Leaf Rust (Puccinia triticina)",
        "category": "Fungal Rust Disease",
        "description": "Small, round to oval bright orange-brown pustules scattered irregularly across leaf blades, bursting to release rusty urediniospores.",
        "chemical_control": [
            "Propiconazole 25% EC (Tilt) @ 1 ml/L (500 ml/ha in 500 L water) at the first sight of pustules.",
            "Tebuconazole 25.9% EC @ 1 ml/L or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L."
        ],
        "organic_control": [
            "Foliar spray of sour buttermilk (fermented curd/lassi) @ 10% concentration.",
            "Bio-fungicide spray of Trichoderma viride or Bacillus subtilis @ 5 g/L."
        ],
        "precautions": [
            "Plant rust-resistant wheat cultivars recommended for your region (e.g., HD-2967, DBW-187, PBW-725).",
            "Avoid late sowing which exposes grain-filling stages to rising temperatures and high rust pressure.",
            "Avoid excessive nitrogen fertilizer; maintain balanced N:P:K (120:60:40 kg/ha)."
        ]
    },
    "Wheat black rust": {
        "crop": "Wheat",
        "clean_name": "Black / Stem Rust (Puccinia graminis f. sp. tritici)",
        "category": "Fungal Rust Disease",
        "description": "Dark reddish-brown to black elongated pustules on stems, leaf sheaths, and glumes. Pustules rupture epidermal tissues, severely lodging stems and shriveling grains.",
        "chemical_control": [
            "Immediate spray of Propiconazole 25% EC @ 1 ml/L or Tebuconazole 25.9% EC @ 1 ml/L.",
            "Mancozeb 75% WP @ 2.5 g/L as a protective cover if neighboring areas report stem rust."
        ],
        "organic_control": [
            "Spray copper-based organic formulations or fermented butter milk (10%).",
            "Foliar application of Ampelomyces quisqualis hyperparasite."
        ],
        "precautions": [
            "Eradicate barberry (Berberis vulgaris) bushes near wheat fields (alternate host for sexual cycle).",
            "Grow stem rust resistant varieties with Sr genes.",
            "Ensure proper field drainage and timely harvest upon maturity."
        ]
    },
    "Wheat___Yellow_Rust": {
        "crop": "Wheat",
        "clean_name": "Yellow / Stripe Rust (Puccinia striiformis)",
        "category": "Fungal Rust Disease",
        "description": "Bright lemon-yellow powdery pustules arranged in prominent parallel stripes along leaf veins. Rapidly spreads in cool, foggy, high-humidity weather.",
        "chemical_control": [
            "At first appearance of yellow stripes, immediately spray Propiconazole 25% EC @ 1 ml/L (200 ml/acre).",
            "Tebuconazole 50% + Trifloxystrobin 25% WG (Nativo) @ 0.6 g/L."
        ],
        "organic_control": [
            "Spray fermented sour lassi (10%) mixed with 1% neem oil.",
            "Foliar spray of Pseudomonas fluorescens @ 5 ml/L."
        ],
        "precautions": [
            "Regular surveillance of sub-mountainous and foothills tracts where stripe rust originates early.",
            "Grow stripe-rust resistant cultivars (e.g. PBW-550, HD-3086, DBW-222).",
            "Avoid sowing highly susceptible varieties in cool, humid microclimates."
        ]
    },
    "Wheat leaf blight": {
        "crop": "Wheat",
        "clean_name": "Leaf Blight / Spot Blotch (Bipolaris sorokiniana)",
        "category": "Fungal Blight Disease",
        "description": "Oval, brown necrotic spots with chlorotic yellow halos on lower leaves, coalescing to cause premature leaf drying and grain discoloration.",
        "chemical_control": [
            "Spray Mancozeb 75% WP @ 2.5 g/L or Zineb 75% WP @ 2 g/L.",
            "Propiconazole 25% EC @ 1 ml/L at flag leaf emergence stage."
        ],
        "organic_control": [
            "Seed treatment with Trichoderma harzianum @ 10 g/kg seed.",
            "Foliar spray of 5% aqueous extract of garlic bulbs or neem oil."
        ],
        "precautions": [
            "Seed treatment with Carboxin 37.5% + Thiram 37.5% DS @ 2.5 g/kg seed before sowing.",
            "Practice crop rotation with non-cereal crops to break fungal spore cycle in soil.",
            "Ensure adequate potash and zinc fertilization to boost foliar resistance."
        ]
    },
    "Wheat powdery mildew": {
        "crop": "Wheat",
        "clean_name": "Wheat Powdery Mildew (Blumeria graminis)",
        "category": "Fungal Foliar Disease",
        "description": "White to grayish-white powdery fungal growth on upper surfaces of leaves and stems, turning dull gray with tiny black cleistothecia specks.",
        "chemical_control": [
            "Wettable Sulphur 80% WP @ 3 g/L or Propiconazole 25% EC @ 1 ml/L.",
            "Hexaconazole 5% EC @ 1 ml/L or Difenoconazole 25% EC @ 0.5 ml/L."
        ],
        "organic_control": [
            "Spray baking soda (Sodium bicarbonate) solution @ 5 g/L + soap emulsifier.",
            "Foliar application of Ampelomyces quisqualis bio-parasite @ 5 g/L."
        ],
        "precautions": [
            "Avoid dense seeding; ensure adequate row spacing (20-22.5 cm) for canopy ventilation.",
            "Avoid excessive nitrogenous fertilization in shaded field areas.",
            "Destroy volunteer wheat plants between growing seasons."
        ]
    },
    "Wheat scab": {
        "crop": "Wheat",
        "clean_name": "Fusarium Head Blight / Scab",
        "category": "Fungal Spike Disease",
        "description": "Premature bleaching of spikelets on wheat heads with pink/orange fungal growth at the base of glumes; grains turn chalky white and shriveled with mycotoxin contamination.",
        "chemical_control": [
            "Spray Tebuconazole 25.9% EC @ 1 ml/L or Metconazole at early flowering (anthesis).",
            "Prothioconazole + Tebuconazole fungicide combination during 50% head emergence."
        ],
        "organic_control": [
            "Bio-fungicide spray of Bacillus subtilis or Trichoderma viride at heading.",
            "Ensure rapid grain drying below 12% moisture immediately post-harvest."
        ],
        "precautions": [
            "Avoid planting wheat directly after maize in minimum-till systems (maize stubble is primary Fusarium reservoir).",
            "Bury crop residues deeply with mouldboard plough before sowing.",
            "Select scab-tolerant varieties with compact flowering windows."
        ]
    },
    "Flag Smut": {
        "crop": "Wheat",
        "clean_name": "Flag Smut of Wheat (Urocystis agropyri)",
        "category": "Fungal Smut Disease",
        "description": "Long dark gray to black stripes running parallel along leaf blades and sheaths, which rupture to release masses of black powdery teliospores, causing leaf twisting and stunted spikes.",
        "chemical_control": [
            "Seed dressing with Carboxin 75% WP @ 2.5 g/kg seed or Tebuconazole 2% DS @ 1.5 g/kg seed.",
            "Foliar spray of Propiconazole 25% EC @ 1 ml/L if symptoms appear on early tillers."
        ],
        "organic_control": [
            "Seed bio-treatment with Trichoderma harzianum @ 10 g/kg seed.",
            "Solar seed treatment: Soak seeds in water for 4 hours, then spread on black polythene in hot summer sun for 4 hours."
        ],
        "precautions": [
            "Strict seed sanitation: Never use seed harvested from previously smut-infected fields.",
            "Practice shallow sowing in warm soil to encourage rapid seedling emergence ahead of fungal spore infection.",
            "Rotate wheat with legumes or brassica crops for at least 2 seasons."
        ]
    },
    "Wheat aphid": {
        "crop": "Wheat",
        "clean_name": "Wheat Aphid (Rhopalosiphum padi / Sitobion avenae)",
        "category": "Sap-Sucking Pest",
        "description": "Green or brownish aphids clustering inside wheat ears and on flag leaves during grain-filling, sucking sap and causing premature drying and reduced test weight.",
        "chemical_control": [
            "At ETL (5 aphids/earhead): Spray Dimethoate 30% EC @ 1.5 ml/L or Thiamethoxam 25% WG @ 0.25 g/L.",
            "Imidacloprid 17.8% SL @ 0.3 ml/L."
        ],
        "organic_control": [
            "Foliar spray of 5% Neem Seed Kernel Extract (NSKE).",
            "Conserve natural predators like Coccinellid beetles (ladybird beetles) and syrphid fly larvae."
        ],
        "precautions": [
            "Install yellow sticky traps @ 10 per acre along field edges.",
            "Avoid late sowing of wheat which coincides with high aphid temperature windows in spring.",
            "Avoid over-application of nitrogenous fertilizers."
        ]
    },
    "Wheat mite": {
        "crop": "Wheat",
        "clean_name": "Brown Wheat Mite (Petrobia latens)",
        "category": "Acarine Foliar Pest",
        "description": "Microscopic dark brown to black mites with yellow-orange legs scraping chlorophyll, giving leaves a bleached, bronze, or scorched appearance in dry rainfed tracts.",
        "chemical_control": [
            "Spray Wettable Sulphur 80% WP @ 3 g/L or Propargite 57% EC @ 2 ml/L.",
            "Fenazaquin 10% EC @ 1.5 ml/L in severely stressed rainfed fields."
        ],
        "organic_control": [
            "Neem oil spray (3 ml/L) mixed with mild soap solution.",
            "Light sprinkler irrigation (brown wheat mites cannot tolerate free moisture and drop off)."
        ],
        "precautions": [
            "Apply a light irrigation during prolonged winter dry spells (mites thrive under dry soil conditions).",
            "Deep summer ploughing to expose diapausing eggs buried in the soil.",
            "Maintain balanced soil nutrition including secondary nutrients."
        ]
    },
    "Wheat Stem fly": {
        "crop": "Wheat",
        "clean_name": "Wheat Stem Fly / Shoot Fly (Atherigona soccata)",
        "category": "Dipteran Stem Borer Pest",
        "description": "Maggots bore into the central growing shoot of young wheat seedlings, causing rotting of growing points and characteristic 'dead heart' symptoms.",
        "chemical_control": [
            "Seed treatment with Imidacloprid 70% WS @ 5 g/kg seed or Thiamethoxam 30% FS @ 4 ml/kg seed.",
            "Spray Chlorpyrifos 20% EC @ 2 ml/L during seedling stage if dead hearts exceed 5%."
        ],
        "organic_control": [
            "Deploy fish meal traps @ 10/acre to attract and trap adult shoot flies.",
            "Neem cake soil application @ 100 kg/acre at sowing."
        ],
        "precautions": [
            "Complete wheat sowing within the recommended window (avoid late November sowing).",
            "Increase seed rate by 10-15% in shoot fly prone tracts to offset seedling mortality.",
            "Pull out and destroy wilted dead heart tillers manually."
        ]
    },

    # ==================== RICE / PADDY ====================
    "Becterial Blight in Rice": {
        "crop": "Rice",
        "clean_name": "Bacterial Leaf Blight (BLB - Xanthomonas oryzae)",
        "category": "Bacterial Foliar Disease",
        "description": "Water-soaked stripes starting from leaf margins and tips, turning wavy yellow, then straw-colored and drying into paper-white lesions with bacterial ooze beads.",
        "chemical_control": [
            "Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline @ 100 mg/L (15 g in 150 L water/acre).",
            "Copper Hydroxide 53.8% DF @ 2 g/L during early leaf margin yellowing."
        ],
        "organic_control": [
            "Spray Pseudomonas fluorescens @ 10 g/L or Bacillus subtilis @ 5 g/L.",
            "Spray 20% fresh cow dung slurry supernatant (centrifuged/strained) on paddy foliage."
        ],
        "precautions": [
            "Drain excess standing water from the field for 3-4 days to arrest bacterial multiplication.",
            "Strictly postpone urea top-dressing until disease progression halts.",
            "Avoid clipping seedling leaf tips during manual transplanting."
        ]
    },
    "Brownspot": {
        "crop": "Rice",
        "clean_name": "Rice Brown Spot (Bipolaris oryzae)",
        "category": "Fungal Foliar Disease",
        "description": "Small circular to oval sesame seed-like brown spots with grayish centers and yellow halos on leaves, stems, and glumes, indicating poor nutrient soil status.",
        "chemical_control": [
            "Mancozeb 75% WP @ 2.5 g/L or Propiconazole 25% EC @ 1 ml/L.",
            "Tebuconazole 25.9% EC @ 1 ml/L or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L."
        ],
        "organic_control": [
            "Bio-seed treatment with Trichoderma viride @ 10 g/kg seed.",
            "Foliar application of Panchagavya (3%) or garlic extract (5%)."
        ],
        "precautions": [
            "Soil application of muriate of potash (MOP) and zinc sulphate to correct underlying nutrient deficiencies.",
            "Avoid water stress; maintain 2-3 cm shallow standing water layer.",
            "Use certified disease-free paddy seed."
        ]
    },
    "Rice Blast": {
        "crop": "Rice",
        "clean_name": "Rice Blast (Magnaporthe oryzae)",
        "category": "Destructive Fungal Disease",
        "description": "Spindle-shaped or eye-shaped lesions with gray/whitish centers and dark reddish-brown margins. Also attacks collar (collar rot) and panicle neck (neck blast), causing panicles to fall over.",
        "chemical_control": [
            "Tricyclazole 75% WP @ 0.6 g/L (120 g/acre) - premier blast fungicide.",
            "Isoprothiolane 40% EC @ 1.5 ml/L or Kasugamycin 3% SL @ 2 ml/L."
        ],
        "organic_control": [
            "Foliar spray of Pseudomonas fluorescens @ 10 g/L at early tillering and panicle initiation.",
            "Spray 5% neem cake extract or fermented sour buttermilk (10%)."
        ],
        "precautions": [
            "Avoid heavy split applications of urea during cloudy, drizzling weather.",
            "Refrain from planting highly blast-susceptible varieties in endemic low-lying basins.",
            "Treat seed with Carbendazim 50% WP @ 2 g/kg seed before nursery sowing."
        ]
    },
    "Leaf smut": {
        "crop": "Rice",
        "clean_name": "Rice Leaf Smut (Entyloma oryzae)",
        "category": "Fungal Foliar Disease",
        "description": "Small, angular, slightly raised black spots (sori) scattered on both leaf surfaces, leading to premature leaf senescence in over-fertilized crops.",
        "chemical_control": [
            "Spray Mancozeb 75% WP @ 2.5 g/L or Copper Oxychloride @ 2.5 g/L.",
            "Propiconazole 25% EC @ 1 ml/L if flag leaf is threatened."
        ],
        "organic_control": [
            "Foliar spray of Trichoderma viride @ 5 g/L.",
            "Spray neem leaf decoction (5%) to slow fungal sporulation."
        ],
        "precautions": [
            "Balance N fertilization with adequate potassium application.",
            "Plow under rice stubbles post-harvest to destroy overwintering chlamydospores.",
            "Maintain weed-free bunds."
        ]
    },
    "Tungro": {
        "crop": "Rice",
        "clean_name": "Rice Tungro Virus (RTV)",
        "category": "Viral Disease (Green Leafhopper Vector)",
        "description": "Transmitted by Green Leafhopper (Nephotettix virescens). Causes severe stunting, reduced tillering, and orange-yellow discoloration beginning from leaf tips.",
        "chemical_control": [
            "Vector control: Spray Dinotefuran 20% SG @ 0.4 g/L or Thiamethoxam 25% WG @ 0.25 g/L.",
            "Imidacloprid 17.8% SL @ 0.3 ml/L targeting green leafhoppers in nursery and main field."
        ],
        "organic_control": [
            "Install light traps @ 1-2 per acre to attract and destroy adult leafhoppers.",
            "Neem oil spray (3%) on leafhoppers to reduce viral transmission efficiency."
        ],
        "precautions": [
            "Rogue out and destroy yellow, stunted plants immediately to remove virus reservoirs.",
            "Synchronize paddy planting in the village tract to prevent leafhopper buildup.",
            "Keep nursery beds covered with insect netting where tungro is endemic."
        ]
    },
    "False Smut in Rice": {
        "crop": "Rice",
        "clean_name": "Rice False Smut (Ustilaginoidea virens)",
        "category": "Fungal Panicle Disease",
        "description": "Transforms individual rice grains into large, velvety yellowish-green to orange spore balls (pseudomorphs) that later turn greenish-black. Causes chalky sterile grains and reduces 1000-grain weight.",
        "chemical_control": [
            "Spray Copper Oxychloride 50 WP @ 2.5 g/L or Propiconazole 25% EC @ 1 ml/L at boot-leaf stage (prior to 50% flowering).",
            "Trifloxystrobin 25% + Tebuconazole 50% WG (Nativo) @ 0.4 g/L at early panicle emergence."
        ],
        "organic_control": [
            "Foliar spray of Trichoderma viride or Pseudomonas fluorescens @ 10 g/L before flowering.",
            "Neem cake soil application (100 kg/acre) to suppress overwintering chlamydospores in soil."
        ],
        "precautions": [
            "Strictly avoid excessive split applications of nitrogenous fertilizers at boot and heading stages.",
            "Collect and destroy smutted panicles in polythene bags to prevent spore balls from shattering into the field.",
            "Use certified smut-free seed stocks and practice hot-water or fungicide seed treatment."
        ]
    },
    "Rice Grain Discoloration": {
        "crop": "Rice",
        "clean_name": "Rice Grain Discoloration Complex",
        "category": "Fungal & Bacterial Panicle Complex",
        "description": "Discoloration, dark brown spotting, or rot on glumes and kernels caused by a complex of Bipolaris, Curvularia, Fusarium, and Sarocladium fungi during humid flowering windows.",
        "chemical_control": [
            "Spray Mancozeb 75 WP @ 2.5 g/L or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L at 50% flowering.",
            "Tebuconazole 25.9% EC @ 1 ml/L for comprehensive glume and kernel protection."
        ],
        "organic_control": [
            "Foliar spray of Pseudomonas fluorescens @ 5 g/L at heading and grain-filling stages.",
            "5% Neem Seed Kernel Extract (NSKE) spray."
        ],
        "precautions": [
            "Avoid excessive late nitrogen fertilization which creates succulent panicle tissues.",
            "Maintain timely field drainage prior to maturity.",
            "Dry harvested grains immediately below 13% moisture content."
        ]
    },

    # ==================== MAIZE / CORN ====================
    "Healthy Maize": {
        "crop": "Maize",
        "clean_name": "Healthy Maize Canopy",
        "category": "Healthy Crop",
        "description": "Broad, arching vibrant green leaves with robust stalk thickness and healthy tassel/silking initiation.",
        "chemical_control": ["No chemical treatments required."],
        "organic_control": ["Foliar spray of liquid bio-NPK consortia to maximize cob weight and kernel fill."],
        "precautions": [
            "Inspect the leaf whorls regularly for early fall armyworm pinholes or frass.",
            "Ensure regular furrow irrigation during tasseling and silking critical moisture periods."
        ]
    },
    "Common_Rust": {
        "crop": "Maize",
        "clean_name": "Common Maize Rust (Puccinia sorghi)",
        "category": "Fungal Rust Disease",
        "description": "Small, powdery, oval to elongate cinnamon-brown pustules scattered over both leaf surfaces, turning brownish-black late in the season.",
        "chemical_control": [
            "Spray Mancozeb 75% WP @ 2.5 g/L or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L.",
            "Propiconazole 25% EC @ 1 ml/L at first appearance of pustules on lower leaves."
        ],
        "organic_control": [
            "Foliar spray of Trichoderma harzianum @ 5 g/L.",
            "Spray fermented sour curd extract (10%) with cow urine."
        ],
        "precautions": [
            "Plant rust-resistant hybrid varieties suited for your agro-ecological zone.",
            "Ensure timely planting: Early planted maize largely escapes heavy rust epidemics.",
            "Destroy crop residues after harvest to minimize fungal survival."
        ]
    },
    "Gray_Leaf_Spot": {
        "crop": "Maize",
        "clean_name": "Gray Leaf Spot (Cercospora zeae-maydis)",
        "category": "Fungal Foliar Disease",
        "description": "Distinctive rectangular, narrow tan-to-gray lesions strictly restricted between leaf veins, causing extensive blighting of upper canopy during warm, overcast periods.",
        "chemical_control": [
            "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L or Pyraclostrobin 20% WG @ 1 g/L.",
            "Propiconazole 25% EC @ 1 ml/L applied at VT (tasseling) stage."
        ],
        "organic_control": [
            "Foliar spray of Pseudomonas fluorescens @ 10 g/L.",
            "Neem oil spray (5 ml/L) as a protective fungal spore deterrent."
        ],
        "precautions": [
            "Rotate maize with non-grass crops (soybean, groundnut, or cotton) for 1-2 years.",
            "Perform conservation tillage that incorporates infected debris to accelerate residue decomposition.",
            "Avoid high plant population densities that trap humid stagnant air in the whorl."
        ]
    },
    "maize ear rot": {
        "crop": "Maize",
        "clean_name": "Maize Ear & Kernel Rot (Fusarium / Gibberella)",
        "category": "Fungal Cob Disease",
        "description": "White, pinkish, or gray cottony fungal mycelium growing on and between kernels, leading to premature bleaching of husks, rotting cobs, and dangerous mycotoxins.",
        "chemical_control": [
            "Foliar spray of Carbendazim 50% WP @ 1 g/L or Mancozeb 75% WP @ 2.5 g/L targeting ear zones during silking.",
            "Tebuconazole 25.9% EC @ 1 ml/L applied during silk emergence."
        ],
        "organic_control": [
            "Bio-seed treatment with Trichoderma viride @ 10 g/kg seed.",
            "Dry harvested cobs rapidly to below 14% moisture before storing."
        ],
        "precautions": [
            "Control ear-infesting pests (corn earworm, fall armyworm) whose feeding wounds allow rot fungi entry.",
            "Harvest promptly when cobs reach physiological maturity; never leave standing in wet fields.",
            "Store grain in clean, moisture-proof, ventilated granaries."
        ]
    },
    "Army worm": {
        "crop": "Maize",
        "clean_name": "Fall Armyworm (Spodoptera frugiperda)",
        "category": "Lepidopteran Invasive Pest",
        "description": "Voracious caterpillars feeding deep within the central whorl, creating ragged 'shot-hole' feeding damage and filling the whorl with sawdust-like wet fecal pellets.",
        "chemical_control": [
            "Chlorantraniliprole 18.5% SC @ 0.4 ml/L directly into the leaf whorl.",
            "Emamectin Benzoate 5% SG @ 0.4 g/L or Spinetoram 11.7% SC @ 0.5 ml/L.",
            "Poison baiting: 10 kg rice bran + 1 kg jaggery fermented 24h + 100g Thiodicarb 75 WP rolled into pellets and placed in whorls."
        ],
        "organic_control": [
            "Install FAW pheromone traps @ 5 per acre for monitoring moth flights.",
            "Spray Nomuraea rileyi or Metarhizium anisopliae @ 5 g/L directly into whorls.",
            "Release egg parasitoid Trichogramma pretiosum @ 50,000/acre."
        ],
        "precautions": [
            "Apply sand or fine wood ash mixed with lime (9:1) directly into the whorl of seedlings (dessicates larvae).",
            "Scout weekly in 'W' pattern across field for early pinhole damage.",
            "Maintain synchronous community planting across neighboring plots."
        ]
    },
    "maize fall armyworm": {
        "crop": "Maize",
        "clean_name": "Fall Armyworm (Spodoptera frugiperda)",
        "category": "Lepidopteran Invasive Pest",
        "description": "Identified by four dark spots in a square on the 8th abdominal segment and inverted 'Y' mark on the head. Destroys whorl leaves and bores into developing ears.",
        "chemical_control": [
            "Chlorantraniliprole 18.5% SC @ 0.4 ml/L or Emamectin Benzoate 5% SG @ 0.4 g/L applied directly into whorls.",
            "Flubendiamide 39.35% SC @ 0.3 ml/L for advanced larval instars."
        ],
        "organic_control": [
            "Pheromone lures @ 5-8 traps/acre.",
            "Spray Bacillus thuringiensis (Bt) aizawai @ 2 g/L during twilight.",
            "Apply neem cake into the whorl."
        ],
        "precautions": [
            "Deep summer ploughing to expose pupae to predatory birds.",
            "Intercrop with desmodium or cowpea (push-pull strategy).",
            "Handpick egg masses covered with hair/scales from leaf undersides."
        ]
    },
    "maize stem borer": {
        "crop": "Maize",
        "clean_name": "Maize Stem Borer (Chilo partellus)",
        "category": "Stem Boring Insect Pest",
        "description": "Larvae feed initially on tender whorl leaves making pinholes, then tunnel into the stem stalk, causing characteristic 'dead heart' and stem breakage.",
        "chemical_control": [
            "Apply Carbofuran 3G @ 3 kg/acre or Fipronil 0.3G granules @ 7 kg/acre into central whorls at 25-30 days after sowing.",
            "Spray Chlorantraniliprole 18.5% SC @ 0.3 ml/L."
        ],
        "organic_control": [
            "Release Trichogramma chilonis @ 60,000/acre at weekly intervals starting 15 days after emergence.",
            "Spray Bacillus thuringiensis @ 2 g/L."
        ],
        "precautions": [
            "Uproot and destroy dead-heart affected plants immediately.",
            "Chop and feed or burn maize stalks post-harvest to kill diapausing larvae.",
            "Intercrop maize with cowpea or lablab beans."
        ]
    },

    # ==================== SUGARCANE ====================
    "Sugarcane Healthy": {
        "crop": "Sugarcane",
        "clean_name": "Healthy Sugarcane Foliage",
        "category": "Healthy Crop",
        "description": "Vigorous, dark green upright canopy with clean midribs and solid, disease-free internode formation.",
        "chemical_control": ["No chemical fungicide or insecticide necessary."],
        "organic_control": ["Soil application of Gluconacetobacter diazotrophicus (nitrogen-fixing endophyte) @ 2 kg/acre."],
        "precautions": [
            "Earthing-up at 90-100 days to prevent lodging and improve soil aeration.",
            "Trash mulching in inter-row spaces to conserve soil moisture and suppress weeds."
        ]
    },
    "RedRot sugarcane": {
        "crop": "Sugarcane",
        "clean_name": "Sugarcane Red Rot (Colletotrichum falcatum)",
        "category": "Devastating Fungal Stalk Disease",
        "description": "The 'cancer' of sugarcane: Causes third/fourth leaf yellowing and drying, internal stalk reddening with characteristic diagnostic horizontal white bands and sour alcoholic odor.",
        "chemical_control": [
            "Sett treatment: Dip setts in Carbendazim 50% WP (1 g/L) + hot water treatment (52°C for 30 mins) before planting.",
            "Foliar spray of Thiophanate Methyl 70% WP @ 1.5 g/L or Carbendazim 50% WP @ 1 g/L."
        ],
        "organic_control": [
            "Soil application of Trichoderma viride enriched pressmud/FYM (5 kg in 1 ton compost/acre).",
            "Sett soaking in Pseudomonas fluorescens @ 10 ml/L for 15 minutes."
        ],
        "precautions": [
            "Immediately rogue out and burn entire affected clumps including root systems; do not ratoon infected crops.",
            "Plant only certified red-rot resistant varieties (e.g. Co-0238, Co-86032, Co-0118).",
            "Ensure strict field drainage; never allow irrigation water to flow from infected fields to healthy plots."
        ]
    },
    "RedRust sugarcane": {
        "crop": "Sugarcane",
        "clean_name": "Sugarcane Red Rust (Puccinia melanocephala)",
        "category": "Fungal Foliar Rust",
        "description": "Elongated, reddish-brown pustules on both leaf surfaces, particularly upper surfaces. Pustules merge into large necrotic patches, leading to leaf shredding and drying.",
        "chemical_control": [
            "Spray Mancozeb 75% WP @ 2.5 g/L or Propiconazole 25% EC @ 1 ml/L.",
            "Tebuconazole 25.9% EC @ 1 ml/L at first notice of rust flecks on young leaves."
        ],
        "organic_control": [
            "Foliar spray of fermented cow urine (10%) mixed with neem oil (3 ml/L).",
            "Spray bio-control agent Trichoderma harzianum @ 5 g/L."
        ],
        "precautions": [
            "Avoid planting highly rust-susceptible cane cultivars.",
            "Detrash lower dried rust-bearing leaves at 150 and 210 days to improve air circulation.",
            "Maintain balanced fertilization with sufficient potassium."
        ]
    },
    "Yellow Rust Sugarcane": {
        "crop": "Sugarcane",
        "clean_name": "Sugarcane Orange / Yellow Rust (Puccinia kuehnii)",
        "category": "Fungal Foliar Rust",
        "description": "Small, yellowish-orange pustules predominantly on lower leaf surfaces, causing yellow-orange tint to the field canopy and premature leaf death.",
        "chemical_control": [
            "Spray Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L.",
            "Propiconazole 25% EC @ 1 ml/L or Chlorothalonil 75% WP @ 2 g/L."
        ],
        "organic_control": [
            "Foliar spray of sour buttermilk (10%).",
            "Application of bio-fungicide Bacillus subtilis @ 5 g/L."
        ],
        "precautions": [
            "Strip and remove lower infected leaves (detrashing) and burn them outside the field.",
            "Avoid excess nitrogen fertilizer which fosters dense, succulent, susceptible foliage.",
            "Ensure wide row spacing (4-5 feet) for better light and air penetration."
        ]
    },
    "Mosaic sugarcane": {
        "crop": "Sugarcane",
        "clean_name": "Sugarcane Mosaic Virus (SCMV)",
        "category": "Viral Disease (Aphid Vector)",
        "description": "Potyvirus transmitted by corn aphids (Rhopalosiphum maidis). Symptoms show striking contrast of dark green islands on pale yellow/chlorotic background along leaf veins.",
        "chemical_control": [
            "No chemical can cure viral infection directly. Control aphid vectors: Spray Dimethoate 30% EC @ 1.5 ml/L or Thiamethoxam 25% WG @ 0.25 g/L.",
            "Imidacloprid 17.8% SL @ 0.3 ml/L on young tillers."
        ],
        "organic_control": [
            "Install yellow sticky traps @ 10 per acre.",
            "Foliar spray of 5% Neem Seed Kernel Extract (NSKE) to deter aphids."
        ],
        "precautions": [
            "Use disease-free seed cane derived from tissue culture (meristem tip culture) or aeroponics.",
            "Rogue out and burn mosaic-stunted clumps promptly.",
            "Do not grow maize, sorghum, or pearl millet adjacent to sugarcane seed plots (they host aphids)."
        ]
    },

    # ==================== DEFAULT FALLBACK ====================
    "default": {
        "crop": "Field Crop",
        "clean_name": "Crop Foliar Pathology",
        "category": "Foliar Pathology",
        "description": "Visual symptoms observed on foliage. Targeted agronomic mitigation and field scouting recommended.",
        "chemical_control": [
            "Broad-spectrum fungicide: Copper Oxychloride 50% WP @ 2.5 g/L or Azoxystrobin @ 1 ml/L.",
            "Consult local agricultural extension center (KVK) for exact chemical registration."
        ],
        "organic_control": [
            "Foliar spray of cold-pressed Neem oil solution (5 ml/L with mild soap emulsifier).",
            "Foliar spray of bio-agent Trichoderma viride or Pseudomonas fluorescens @ 5 g/L."
        ],
        "precautions": [
            "Prune visibly infected foliage and dispose safely away from the cropped acreage.",
            "Ensure balanced fertilization (avoid excessive nitrogen; supplement potassium and micronutrients).",
            "Avoid overhead sprinkler irrigation which disperses foliar pathogens."
        ]
    }
}


class AdvisoryEngine:
    """
    Generates actionable chemical, biological, and cultural agronomic advisories
    tailored to the diagnosed disease, pest detections, and lesion severity percentage.
    """
    @classmethod
    def get_crop_and_condition(cls, raw_label: str) -> Dict[str, str]:
        """
        Parses raw label into clean Crop Name, Clean Pathology Name, and Category.
        """
        # Exact match
        if raw_label in DISEASE_ADVISORY_DB:
            entry = DISEASE_ADVISORY_DB[raw_label]
            return {
                "crop": entry["crop"],
                "clean_name": entry["clean_name"],
                "category": entry["category"]
            }

        # Fuzzy match
        norm_label = raw_label.lower().replace("_", " ").strip()
        for key, entry in DISEASE_ADVISORY_DB.items():
            norm_key = key.lower().replace("_", " ").strip()
            if norm_key in norm_label or norm_label in norm_key:
                return {
                    "crop": entry["crop"],
                    "clean_name": entry["clean_name"],
                    "category": entry["category"]
                }

        # Deduce crop from text
        crop = "Field Crop"
        if "cotton" in norm_label:
            crop = "Cotton"
        elif "wheat" in norm_label:
            crop = "Wheat"
        elif any(k in norm_label for k in ["rice", "brownspot", "tungro", "paddy", "blast", "smut", "false smut", "sheath", "grain"]):
            crop = "Rice"
        elif "maize" in norm_label or "corn" in norm_label or "army worm" in norm_label:
            crop = "Maize"
        elif "sugarcane" in norm_label:
            crop = "Sugarcane"

        clean_name = raw_label.replace("_", " ").title()
        return {
            "crop": crop,
            "clean_name": clean_name,
            "category": "Foliar Condition"
        }

    @classmethod
    def generate_advisory(
        cls,
        disease_name: str,
        confidence: float,
        severity_pct: float,
        pest_count: int = 0,
        pests: List[str] = None,
        resolution_state: Optional[str] = None
    ) -> Dict[str, Any]:
        pests = pests or []
        meta = cls.get_crop_and_condition(disease_name)

        # Match entry in DB
        matched_key = "default"
        if disease_name in DISEASE_ADVISORY_DB:
            matched_key = disease_name
        else:
            norm_disease = disease_name.lower().replace("_", " ").strip()
            for key in DISEASE_ADVISORY_DB:
                norm_key = key.lower().replace("_", " ").strip()
                if norm_key in norm_disease or norm_disease in norm_key:
                    matched_key = key
                    break

        base_advisory = DISEASE_ADVISORY_DB.get(matched_key, DISEASE_ADVISORY_DB["default"]).copy()

        # Adapt urgency based on severity % and category
        if "healthy" in meta["category"].lower():
            urgency = "LOW - Crop is Healthy & Vigorous"
        elif severity_pct >= 30.0:
            urgency = "HIGH - Immediate intervention needed to prevent yield loss"
        elif severity_pct >= 10.0:
            urgency = "MODERATE - Action recommended within 48-72 hours"
        else:
            urgency = "LOW - Early symptoms; preventive bio-control recommended"

        pest_notes = []
        if pest_count > 0:
            pest_notes.append(f"Alert: {pest_count} active pest(s) localized ({', '.join(pests)}). Integrated pest control recommended.")

        # STRICT SCIENTIFIC SAFETY RULE:
        # Disease-specific chemical & organic recommendations are permitted ONLY when
        # resolution is CONSENSUS or CNN_ONLY.
        # For GEMINI_SUSPECTED, CONFLICT, UNKNOWN, INSUFFICIENT_EVIDENCE:
        # Default to NO disease-specific chemical or treatment recommendations,
        # providing only scouting, image-quality, and expert-verification guidance.
        if resolution_state and resolution_state not in ["CONSENSUS", "CNN_ONLY", "CNN_ACCEPTED"]:
            return {
                "disease_identified": meta["clean_name"],
                "raw_disease_name": disease_name,
                "crop": meta["crop"],
                "category": meta["category"],
                "confidence_level": f"{confidence * 100:.1f}%",
                "severity_level": f"{severity_pct:.1f}%",
                "urgency": "ADVISORY - In-Field Verification Required",
                "disease_description": (
                    f"Condition suspected: '{meta['clean_name']}'. "
                    f"Due to resolution state ({resolution_state}), disease-specific chemical interventions "
                    f"are strictly withheld pending physical in-field confirmation."
                ),
                "chemical_control": [],
                "organic_control": [],
                "cultural_practices": [
                    f"Physically inspect the crop canopy and leaf tissue for characteristic symptoms of {meta['clean_name']}.",
                    "Obtain confirmation from a local agricultural extension officer or plant pathologist.",
                    "Capture a sharp, well-lit close-up photograph of the symptom margins for re-evaluation.",
                    "Avoid applying chemical sprays without confirmed diagnosis to prevent unnecessary expense and chemical burn."
                ],
                "pest_alerts": pest_notes
            }

        return {
            "disease_identified": meta["clean_name"],
            "raw_disease_name": disease_name,
            "crop": meta["crop"],
            "category": meta["category"],
            "confidence_level": f"{confidence * 100:.1f}%",
            "severity_level": f"{severity_pct:.1f}%",
            "urgency": urgency,
            "disease_description": base_advisory.get("description", ""),
            "chemical_control": base_advisory.get("chemical_control", []),
            "organic_control": base_advisory.get("organic_control", []),
            "cultural_practices": base_advisory.get("cultural_practices") or base_advisory.get("precautions", []),
            "pest_alerts": pest_notes
        }

