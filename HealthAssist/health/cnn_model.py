import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_disease_model.h5")

# Global model (lazy init)
model = None

# Class names (must match the trained model output)
CLASS_NAMES = [
    "Acne", "Eczema", "Tinea corporis", "Rosacea", "Vitiligo", "Melasma",
    "Urticaria", "Fungal Infection", "Bacterial Infection", "Viral Infection",
    "Scabies", "Seborrheic Dermatitis", "Contact Dermatitis", "Lupus Lesion",
    "Actinic Keratoses", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
    "Melanoma", "Benign Nevus", "Seborrheic Keratosis", "Dermatofibroma",
    "Cherry Angioma", "Vascular Lesion"
]
# Your SKIN_INFO dictionary (paste all your entries here)
SKIN_INFO = {
  "Acne": {
    "description": "Blocked hair follicles cause pimples, blackheads, or cysts.",
    "medical_treatment": "Benzoyl peroxide, retinoids, antibiotics (clindamycin), isotretinoin (severe cases).",
    "home_remedies": "Tea tree oil, gentle cleansing, avoid over-washing.",
    "diet": "Avoid sugar and dairy; eat zinc-rich foods and omega-3s."
  },
  "Eczema": {
    "description": "Chronic itchy, dry, inflamed skin patches, often on elbows, knees, or face.",
    "medical_treatment": "Corticosteroid creams, antihistamines, moisturizers, immunosuppressants.",
    "home_remedies": "Oatmeal baths, coconut oil, wear soft cotton clothing.",
    "diet": "Avoid triggers (dairy, nuts, eggs in some cases); anti-inflammatory foods."
  },
  "Tinea corporis": {
    "description": "Autoimmune disease causing red plaques with silvery scales.",
    "medical_treatment": "Topical steroids, vitamin D creams, biologics.",
    "home_remedies": "Aloe vera, Dead Sea salt baths, stress reduction.",
    "diet": "Avoid processed foods & alcohol; include omega-3s and greens."
  },
  "Rosacea": {
    "description": "Chronic facial redness & flushing; may have papules/pustules.",
    "medical_treatment": "Topical metronidazole, azelaic acid, oral doxycycline.",
    "home_remedies": "Cold compress, green tea soaks; trigger avoidance.",
    "diet": "Avoid spicy foods, alcohol, hot drinks; calming, hydrating foods."
  },
  "Vitiligo": {
    "description": "Loss of skin pigment (melanin) causing well-defined white patches.",
    "medical_treatment": "Topical steroids/tacrolimus, phototherapy, grafts.",
    "home_remedies": "Broad-spectrum sunscreen; cosmetic camouflage.",
    "diet": "Copper/B12/folate-rich foods; minimize ultra-processed foods."
  },
  "Melasma": {
    "description": "Brown patches on face; worsens with sun/hormones.",
    "medical_treatment": "Hydroquinone, triple combo creams, peels, laser.",
    "home_remedies": "Aloe vera, strict photoprotection.",
    "diet": "Antioxidant-rich foods: berries, citrus, green tea."
  },
  "Urticaria": {
    "description": "Transient, itchy wheals; often allergic or idiopathic.",
    "medical_treatment": "Second-generation antihistamines; short steroid burst if severe.",
    "home_remedies": "Cool compress, loose clothing.",
    "diet": "Avoid known triggers; low-histamine trial if advised."
  },
  "Fungal Infection": {
    "description": "Tinea/candidiasis; itchy, red, often annular lesions.",
    "medical_treatment": "Topical clotrimazole/terbinafine; oral azoles if extensive.",
    "home_remedies": "Keep areas dry; dilute tea tree oil for tinea pedis.",
    "diet": "Cut excess sugar; add probiotics (yogurt, kefir)."
  },
  "Bacterial Infection": {
    "description": "Impetigo/cellulitis; erythema, crusting or warmth & swelling.",
    "medical_treatment": "Topical mupirocin; oral antibiotics for cellulitis.",
    "home_remedies": "Honey (antibacterial) adjunct, not replacement.",
    "diet": "Immune-support foods: garlic, ginger, citrus."
  },
  "Viral Infection": {
    "description": "Herpes/warts/molluscum; blisters, warty papules or umbilicated bumps.",
    "medical_treatment": "Acyclovir for HSV; cryo/cantharidin for warts.",
    "home_remedies": "Aloe/calamine for comfort.",
    "diet": "Vitamin C foods; lysine-rich foods may help HSV."
  },
  "Scabies": {
    "description": "Sarcoptes mite; intense nocturnal itch; burrows in webs of fingers, wrists.",
    "medical_treatment": "Permethrin 5% whole-body; oral ivermectin if needed; treat contacts.",
    "home_remedies": "Wash bedding/clothes hot cycle; bag items 3+ days.",
    "diet": "General immune-support diet."
  },
  "Seborrheic Dermatitis": {
    "description": "Scalp/face erythema with greasy scale; Malassezia-related.",
    "medical_treatment": "Ketoconazole/zinc pyrithione shampoos; mild topical steroids.",
    "home_remedies": "Coconut oil, diluted ACV rinses.",
    "diet": "Reduce sugar; include omega-3s."
  },
  "Contact Dermatitis": {
    "description": "Allergic/irritant reaction to allergens/chemicals/metals.",
    "medical_treatment": "Topical steroids; antihistamines for itch; avoidant strategy.",
    "home_remedies": "Cool compresses, colloidal oatmeal.",
    "diet": "Anti-inflammatory pattern; eliminate trigger if food-related."
  },
  "Lupus Lesion": {
    "description": "Photosensitive malar/discoid rashes; autoimmune.",
    "medical_treatment": "Photoprotection, topical steroids/calcineurin inhibitors; hydroxychloroquine.",
    "home_remedies": "Sun avoidance, stress management.",
    "diet": "Anti-inflammatory foods; omega-3s."
  },
  "Actinic Keratoses": {
    "description": "Rough scaly macules on sun-damaged skin; precancerous.",
    "medical_treatment": "Cryotherapy; 5-FU/imiquimod; photodynamic therapy.",
    "home_remedies": "Sun protection.",
    "diet": "Carotenoid/antioxidant-rich produce."
  },
  "Basal Cell Carcinoma": {
    "description": "Most common skin cancer; pearly papule; rarely metastasizes.",
    "medical_treatment": "Excision/Mohs; topical imiquimod in select cases.",
    "home_remedies": "None—seek medical care; photoprotection.",
    "diet": "Plant-forward, antioxidants."
  },
  "Squamous Cell Carcinoma": {
    "description": "Keratinizing tumor; scaly/red nodule/ulcer; can metastasize.",
    "medical_treatment": "Excision/Mohs; radiation in select cases.",
    "home_remedies": "None—medical care essential; photoprotection.",
    "diet": "Tomatoes (lycopene), green tea; avoid smoking."
  },
  "Melanoma": {
    "description": "Aggressive melanoma; evolving asymmetric pigmented lesion.",
    "medical_treatment": "Wide excision; immunotherapy/targeted therapy.",
    "home_remedies": "None—urgent specialist care.",
    "diet": "Antioxidant-rich; avoid alcohol/smoking."
  },
  "Benign Nevus": {
    "description": "Common mole; benign melanocytic nevus.",
    "medical_treatment": "Removal only if symptomatic or changing.",
    "home_remedies": "None needed.",
    "diet": "General healthy diet."
  },
  "Seborrheic Keratosis": {
    "description": "Stuck-on, waxy papules; benign.",
    "medical_treatment": "Cryotherapy/curettage/laser if cosmetic.",
    "home_remedies": "Not required.",
    "diet": "Not diet-related."
  },
  "Dermatofibroma": {
    "description": "Firm dermal nodule; dimples on pinching.",
    "medical_treatment": "Excision if painful/cosmetic.",
    "home_remedies": "None.",
    "diet": "Balanced diet."
  },
  "Cherry Angioma": {
    "description": "Benign red vascular papules.",
    "medical_treatment": "Laser/electrocautery if desired.",
    "home_remedies": "None required.",
    "diet": "Balanced diet."
  },
  "Vascular Lesion": {
    "description": "Capillary/vascular malformations or angiomas.",
    "medical_treatment": "Laser therapy depending on type.",
    "home_remedies": "Photoprotection for visible areas.",
    "diet": "General healthy diet."
  }
}

def get_model():
    """Load model only once (lazy loading)."""
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model


def predict_skin_disease(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Get model and make prediction
    model = get_model()
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    class_name = CLASS_NAMES[idx]
    confidence = float(np.max(preds[0]) * 100)

    # Get detailed info
    info = SKIN_INFO.get(class_name, {})
    return {
        "class_name": class_name,
        "confidence": confidence,
        "description": info.get("description", ""),
        "medical_treatment": info.get("medical_treatment", ""),
        "home_remedies": info.get("home_remedies", ""),
        "diet": info.get("diet", "")
    }
