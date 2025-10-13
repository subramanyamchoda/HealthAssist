import os
import json
import base64
import requests
import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.contrib.auth import authenticate
from rest_framework import status, serializers, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import your models and serializers
from .models import UserProfile, HealthRecord, SkinDisease
from .serializers import (
    UserProfileSerializer,
    HealthRecordSerializer,
    SkinDiseaseSerializer,
)

from django.conf import settings
# -------------------- Serializer --------------------




class WelcomeView(APIView):
    def get(self, request):
        return Response({"message": "👋 Welcome to HealthAssist API"})


class UserRegister(APIView):
    def post(self, request):
        serializer = UserProfileSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLogin(APIView):
    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        try:
            user = UserProfile.objects.get(username=username)
            if user.password == password:
                request.session["user_id"] = user.id
                serializer = UserProfileSerializer(user)
                return Response({"user": serializer.data}, status=status.HTTP_200_OK)
            return Response({"error": "Invalid password"}, status=status.HTTP_401_UNAUTHORIZED)
        except UserProfile.DoesNotExist:
            return Response({"error": "Invalid username"}, status=status.HTTP_401_UNAUTHORIZED)


class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer


class UserInfo(APIView):
    def get(self, request):
        user_id = request.session.get("user_id")
        if not user_id:
            return Response({"error": "User not logged in"}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = UserProfile.objects.get(id=user_id)
            serializer = UserProfileSerializer(user)
            return Response({"user": serializer.data}, status=status.HTTP_200_OK)
        except UserProfile.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

class HealthRecordView(APIView):
    def post(self, request):
        user_id = request.data.get("user_id")
        if not user_id:
            return Response({"error": "Missing user ID"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = UserProfile.objects.get(id=user_id)
        except UserProfile.DoesNotExist:
            return Response({"error": "Invalid user ID"}, status=status.HTTP_404_NOT_FOUND)

        user_message = request.data.get("message", "")
        uploaded_image = request.FILES.get("image")
        city_name = user.address or "Ongole"

        fallback_advice = (
            "⚠️ The AI assistant is currently unavailable.\n\n"
            "General suggestions:\n"
            "1. Rest and drink plenty of water.\n"
            "2. Use paracetamol for fever.\n"
            "3. Try simple home remedies like honey with warm water for cough.\n"
            "4. See a doctor if severe symptoms appear.\n\n"
            "⚠️ This is not medical advice. Please consult a doctor."
        )

        bot_reply = fallback_advice
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

        if groq_api_key:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful health assistant. "
                            "Always answer in very simple English with 4 parts:\n"
                            "1. Cause explanation\n2. Home remedies\n3. Safe OTC medicines\n"
                            "4. Advice to see a doctor if it worsens."
                        ),
                    }
                ]

                # ✅ If image provided, send as base64 to Groq
                if uploaded_image:
                    image_bytes = uploaded_image.read()
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    messages.append(
                        {"role": "user", "content": f"Skin image (base64): {image_b64}"}
                    )
                else:
                    # Otherwise use text message
                    messages.append({"role": "user", "content": f"My symptoms: {user_message}"})

                groq_res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": messages,
                        "max_tokens": 400,
                        "temperature": 0.6,
                    },
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                groq_res.raise_for_status()
                choices = groq_res.json().get("choices", [])
                if choices and "message" in choices[0]:
                    bot_reply = choices[0]["message"].get("content", fallback_advice).strip()
            except requests.RequestException as e:
                print("Groq API error:", str(e))
                bot_reply = fallback_advice

        # --- Geoapify hospital search ---
        hospitals = []
        geoapify_api_key = os.environ.get("GEOAPIFY_API_KEY", "")
        if geoapify_api_key:
            try:
                geo_res = requests.get(
                    "https://api.geoapify.com/v1/geocode/search",
                    params={"text": city_name, "apiKey": geoapify_api_key},
                    timeout=10,
                )
                geo_res.raise_for_status()
                features = geo_res.json().get("features", [])
                if features:
                    lat, lon = features[0]["geometry"]["coordinates"][1], features[0]["geometry"]["coordinates"][0]
                    places_res = requests.get(
                        "https://api.geoapify.com/v2/places",
                        params={
                            "categories": "healthcare.hospital",
                            "bias": f"proximity:{lon},{lat}",
                            "limit": 5,
                            "apiKey": geoapify_api_key,
                        },
                        timeout=10,
                    )
                    places_res.raise_for_status()
                    hospitals = [
                        {
                            "name": f["properties"].get("name", "Unnamed"),
                            "address": f["properties"].get("formatted", "Address not available"),
                            "lat": f["properties"].get("lat"),
                            "lon": f["properties"].get("lon"),
                            "map_link": f"https://www.google.com/maps/search/?api=1&query={f['properties'].get('lat')},{f['properties'].get('lon')}"
                        }
                        for f in places_res.json().get("features", [])
                    ]
            except requests.RequestException:
                hospitals = [{"error": "Unable to fetch hospitals at this time."}]

        
        record = HealthRecord.objects.create(
            user=user,
            message=user_message,
            bot_response=bot_reply
        )

        return Response(
            {
                "record": HealthRecordSerializer(record).data,
                "suggested_hospitals": hospitals,
            },
            status=status.HTTP_201_CREATED
        )



import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from .models import SkinDisease, UserProfile
from .serializers import SkinDiseaseSerializer

# ---------------- Load Model and Class Data ----------------
MODEL_PATH = os.path.join("health", "skin_disease_model.h5")
JSON_PATH = os.path.join("health", "class_indices.json")

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load class data
try:
    with open(JSON_PATH, "r") as f:
        class_data = json.load(f)
except Exception as e:
    print(f"⚠️ Error loading class_indices.json: {e}")
    class_data = {}

# Map model indices → class names
if class_data:
    first_val = list(class_data.values())[0]
    if isinstance(first_val, int):
        # Simple mapping: {"Acne": 0, "Eczema": 1}
        classes = {v: k for k, v in class_data.items()}
    elif isinstance(first_val, dict) and "index" in first_val:
        # Detailed mapping: {"Acne": {"index":0, ...}}
        classes = {v["index"]: k for k, v in class_data.items()}
    else:
        classes = {}
else:
    classes = {}

# ---------------- Prediction View ----------------
class SkinDiseasePredictionView(APIView):
    """
    POST: Predict skin disease from uploaded image.
    Expects: user_id, image (multipart/form-data)
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        # ✅ Get user_id from request
        user_id = request.data.get("user_id")
        if not user_id:
            return Response({"error": "User ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # ✅ Validate user
        try:
            user = UserProfile.objects.get(id=user_id)
        except UserProfile.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # ✅ Validate uploaded image
        serializer = SkinDiseaseSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = serializer.validated_data["image"]
        temp_path = default_storage.save(f"temp/{image_file.name}", image_file)
        full_path = default_storage.path(temp_path)

        try:
            if model is None:
                raise ValueError("❌ Model not loaded. Please check model path.")

            # ---------------- Preprocess Image ----------------
            img = image.load_img(full_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ---------------- Predict ----------------
            preds = model.predict(img_array)
            predicted_index = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            predicted_label = classes.get(predicted_index, "Unknown")

            # ---------------- Get Disease Info ----------------
            disease_info = class_data.get(predicted_label, {})
            if isinstance(disease_info, int):
                disease_info = {"index": disease_info}

            for key in ["description", "medical_treatment", "home_remedies", "diet", "specialist_doctors"]:
                disease_info.setdefault(key, "" if key != "specialist_doctors" else [])

            # ---------------- Save Prediction ----------------
            record = SkinDisease.objects.create(
                user=user,
                image=image_file,
                bot_response=predicted_label,
            )

            # ---------------- Build Response ----------------
            result = {
                "id": record.id,
                "user": user.username,
                "class_name": predicted_label,
                "confidence": round(confidence * 100, 2),
                "description": disease_info.get("description"),
                "medical_treatment": disease_info.get("medical_treatment"),
                "home_remedies": disease_info.get("home_remedies"),
                "diet": disease_info.get("diet"),
                "specialist_doctors": disease_info.get("specialist_doctors"),
            }

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            default_storage.delete(temp_path)

        return Response(result, status=status.HTTP_200_OK)
