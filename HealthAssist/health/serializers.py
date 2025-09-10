from rest_framework import serializers
from .models import UserProfile, HealthRecord

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = '__all__'


class HealthRecordSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    bot_reply = serializers.CharField(source="bot_response", read_only=True)  # explicit field

    class Meta:
        model = HealthRecord
        fields = ["id", "user", "message", "bot_response", "bot_reply"]

class SkinDiseaseSerializer(serializers.Serializer):
    image = serializers.ImageField()