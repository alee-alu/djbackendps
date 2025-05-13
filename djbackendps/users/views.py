from django.contrib.auth.models import User
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from django.contrib.auth import authenticate

from .serializers import UserSerializer, RegisterSerializer

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        print(f"Registration request received: {request.data}")

        try:
            # Create the serializer with the request data
            serializer = self.get_serializer(data=request.data)

            # Validate the serializer data
            try:
                if serializer.is_valid():
                    print("Serializer validation passed")
                else:
                    print(f"Serializer validation failed: {serializer.errors}")
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                import traceback
                print(f"Error during serializer validation: {e}")
                print(traceback.format_exc())
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

            # Save the user
            try:
                user = serializer.save()
                print(f"User saved successfully: {user.username}, ID: {user.id}")
            except Exception as e:
                import traceback
                print(f"Error saving user: {e}")
                print(traceback.format_exc())
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

            # Create token for the new user
            try:
                token, created = Token.objects.get_or_create(user=user)
                print(f"Token created for user {user.username}: {token.key}")
            except Exception as e:
                import traceback
                print(f"Error creating token: {e}")
                print(traceback.format_exc())
                return Response({"error": f"User created but token generation failed: {str(e)}"},
                               status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Return the response
            return Response({
                "user": UserSerializer(user, context=self.get_serializer_context()).data,
                "token": token.key,
                "message": "User registered successfully"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            import traceback
            print(f"Unexpected error during registration: {e}")
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoginView(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        password = request.data.get('password')

        user = authenticate(username=username, password=password)

        if user:
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user_id': user.pk,
                'username': user.username,
                'email': user.email,
                'role': user.profile.role
            })
        else:
            return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

class UserDetailView(generics.RetrieveAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user

class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        # Delete the token to logout
        try:
            request.user.auth_token.delete()
            return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
