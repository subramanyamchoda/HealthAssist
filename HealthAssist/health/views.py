import os
import requests
from django.contrib.auth import authenticate
from rest_framework import status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
import base64
from .models import UserProfile, HealthRecord
from .serializers import UserProfileSerializer, HealthRecordSerializer
import base64
from .serializers import SkinDiseaseSerializer
from .cnn_model import predict_skin_disease




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
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from .cnn_model import predict_skin_disease

class SkinDiseasePredictionView(APIView):
    """
    Predict skin disease from an uploaded image and return detailed info.
    """
    def post(self, request, format=None):
        if 'image' not in request.FILES:
            return Response({"error": "No image uploaded"}, status=400)

        # Save uploaded image temporarily
        image_file = request.FILES['image']
        path = default_storage.save(f"temp/{image_file.name}", image_file)
        full_path = default_storage.path(path)

        try:
            # Get prediction
            result = predict_skin_disease(full_path)
        finally:
            # Delete temporary file
            default_storage.delete(path)

        return Response(result)
