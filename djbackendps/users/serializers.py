from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from .models import UserProfile

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['role', 'date_joined', 'last_login']

class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'profile']
        read_only_fields = ['id']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    role = serializers.ChoiceField(choices=UserProfile.USER_ROLES, default='user')

    class Meta:
        model = User
        fields = ['username', 'password', 'password2', 'email', 'first_name', 'last_name', 'role']
        extra_kwargs = {
            'first_name': {'required': True},
            'last_name': {'required': True},
            'email': {'required': True}
        }

    def validate(self, attrs):
        print(f"Validating registration data: {attrs}")

        # Check if passwords match
        if attrs['password'] != attrs['password2']:
            print("Password validation failed: passwords don't match")
            raise serializers.ValidationError({"password": "Password fields didn't match."})

        # Check if username already exists
        username = attrs.get('username')
        if User.objects.filter(username=username).exists():
            print(f"Username validation failed: {username} already exists")
            raise serializers.ValidationError({"username": f"User with username '{username}' already exists."})

        # Check if email already exists
        email = attrs.get('email')
        if User.objects.filter(email=email).exists():
            print(f"Email validation failed: {email} already exists")
            raise serializers.ValidationError({"email": f"User with email '{email}' already exists."})

        print("Registration data validation successful")
        return attrs

    def create(self, validated_data):
        print(f"Creating user with data: {validated_data}")

        try:
            role = validated_data.pop('role')
            password2 = validated_data.pop('password2')

            user = User.objects.create(
                username=validated_data['username'],
                email=validated_data['email'],
                first_name=validated_data['first_name'],
                last_name=validated_data['last_name']
            )

            user.set_password(validated_data['password'])
            user.save()

            # Set the role in the user profile
            user.profile.role = role
            user.profile.save()

            print(f"User created successfully: {user.username}, ID: {user.id}")
            return user
        except Exception as e:
            import traceback
            print(f"Error creating user: {e}")
            print(traceback.format_exc())
            raise
