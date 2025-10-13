from rest_framework import serializers
from .models import UserProfile, HealthRecord,SkinDisease

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = '__all__'


class HealthRecordSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    bot_reply = serializers.CharField(source="bot_response", read_only=True)  

    class Meta:
        model = HealthRecord
        fields = ["id", "user", "message", "bot_response", "bot_reply"]


class SkinDiseaseSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    bot_response = serializers.CharField(read_only=True)

    class Meta:
        model = SkinDisease
        fields = ["id", "user", "image", "bot_response"]