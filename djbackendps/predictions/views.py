from rest_framework import viewsets, status, permissions
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from .models import PredictionRecord
from .serializers import PredictionSerializer

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = PredictionRecord.objects.all()
    serializer_class = PredictionSerializer
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        # Extract data from request
        data = request.data
        prediction_type = data.get('prediction_type')
        prediction_data = data.get('prediction_data', {})
        prediction_result = data.get('prediction_result', '')
        risk_score = data.get('risk_score', 0.0)

        # Create a record based on prediction type
        # Handle different formats of prediction_result
        if isinstance(prediction_result, bool):
            result_bool = prediction_result
        elif isinstance(prediction_result, str):
            result_bool = 'has' in prediction_result.lower() or 'yes' in prediction_result.lower() or 'true' in prediction_result.lower() or '1' in prediction_result
        else:
            result_bool = bool(prediction_result)

        # Add model_type field
        model_type = data.get('model_type', prediction_type)

        # Get common fields
        username = data.get('username')
        age = prediction_data.get('age')
        gender = prediction_data.get('gender')

        # Convert risk_score to float
        try:
            risk_score_float = float(risk_score) if risk_score is not None else 0.0
        except (ValueError, TypeError):
            risk_score_float = 0.0

        # Convert age to integer
        try:
            age_int = int(age) if age is not None else None
        except (ValueError, TypeError):
            age_int = None

        record_data = {
            'disease_type': prediction_type,
            'prediction_result': result_bool,  # Convert to boolean
            'risk_score': risk_score_float,
            'username': username,  # Include username field now that it exists in the database
            'age': age_int,
            'gender': gender,
            'model_type': model_type,  # Include model_type field
        }

        # Print common fields for debugging
        print(f"Common fields: disease_type={prediction_type}, prediction_result={result_bool}, risk_score={risk_score_float}, username={username}, age={age_int}, gender={gender}, model_type={model_type}")

        # Add type-specific fields
        if prediction_type == 'diabetes':
            # Get diabetes-specific fields
            pregnancies = prediction_data.get('pregnancies')
            glucose = prediction_data.get('glucose')
            blood_pressure = prediction_data.get('blood_pressure')
            skin_thickness = prediction_data.get('skin_thickness')
            insulin = prediction_data.get('insulin')
            bmi = prediction_data.get('bmi')
            pedigree = prediction_data.get('pedigree')

            # Convert string values to appropriate types
            try:
                # Convert numeric values
                pregnancies_val = int(pregnancies) if pregnancies is not None else None
                glucose_val = float(glucose) if glucose is not None else None
                blood_pressure_val = float(blood_pressure) if blood_pressure is not None else None
                skin_thickness_val = float(skin_thickness) if skin_thickness is not None else None
                insulin_val = float(insulin) if insulin is not None else None
                bmi_val = float(bmi) if bmi is not None else None
                diabetes_pedigree_val = float(pedigree) if pedigree is not None else None

                # Update record data with converted values
                record_data.update({
                    'pregnancies': pregnancies_val,
                    'glucose': glucose_val,
                    'blood_pressure': blood_pressure_val,
                    'skin_thickness': skin_thickness_val,
                    'insulin': insulin_val,
                    'bmi': bmi_val,
                    'diabetes_pedigree': diabetes_pedigree_val,
                })

                # Print the converted values for debugging
                print(f"Converted diabetes fields: {record_data}")

            except Exception as e:
                print(f"Error converting diabetes fields: {e}")
                # If conversion fails, try with original values
                record_data.update({
                    'pregnancies': pregnancies,
                    'glucose': glucose,
                    'blood_pressure': blood_pressure,
                    'skin_thickness': skin_thickness,
                    'insulin': insulin,
                    'bmi': bmi,
                    'diabetes_pedigree': pedigree,
                })
        elif prediction_type == 'heart':
            # Get heart-specific fields with proper type conversion
            cp = prediction_data.get('cp')
            trestbps = prediction_data.get('trestbps')
            chol = prediction_data.get('chol')
            fbs = prediction_data.get('fbs')
            restecg = prediction_data.get('restecg')
            thalach = prediction_data.get('thalach')
            exang = prediction_data.get('exang')
            oldpeak = prediction_data.get('oldpeak')
            slope = prediction_data.get('slope')
            ca = prediction_data.get('ca')
            thal = prediction_data.get('thal')

            # Convert string values to appropriate types
            try:
                # Convert chest_pain_type to integer
                if isinstance(cp, str) and '(' in cp:
                    cp = int(cp.split('(')[1].split(')')[0])
                chest_pain_type = int(cp) if cp is not None else None

                # Convert numeric values to float
                resting_bp = float(trestbps) if trestbps is not None else None
                cholesterol = float(chol) if chol is not None else None

                # Convert fasting_blood_sugar to boolean
                if isinstance(fbs, str):
                    fasting_blood_sugar = fbs.lower() == 'yes'
                else:
                    fasting_blood_sugar = bool(fbs) if fbs is not None else None

                # Convert rest_ecg to integer
                if isinstance(restecg, str) and '(' in restecg:
                    restecg = int(restecg.split('(')[1].split(')')[0])
                rest_ecg = int(restecg) if restecg is not None else None

                # Convert max_heart_rate to float
                max_heart_rate = float(thalach) if thalach is not None else None

                # Convert exercise_induced_angina to boolean
                if isinstance(exang, str):
                    exercise_induced_angina = exang.lower() == 'yes'
                else:
                    exercise_induced_angina = bool(exang) if exang is not None else None

                # Convert st_depression to float
                st_depression = float(oldpeak) if oldpeak is not None else None

                # Convert st_slope to integer
                if isinstance(slope, str) and '(' in slope:
                    slope = int(slope.split('(')[1].split(')')[0])
                st_slope = int(slope) if slope is not None else None

                # Convert num_major_vessels to integer
                num_major_vessels = int(ca) if ca is not None else None

                # Convert thalassemia to integer
                if isinstance(thal, str) and '(' in thal:
                    thal = int(thal.split('(')[1].split(')')[0])
                thalassemia = int(thal) if thal is not None else None

                # Update record data with converted values
                record_data.update({
                    'chest_pain_type': chest_pain_type,
                    'resting_bp': resting_bp,
                    'cholesterol': cholesterol,
                    'fasting_blood_sugar': fasting_blood_sugar,
                    'rest_ecg': rest_ecg,
                    'max_heart_rate': max_heart_rate,
                    'exercise_induced_angina': exercise_induced_angina,
                    'st_depression': st_depression,
                    'st_slope': st_slope,
                    'num_major_vessels': num_major_vessels,
                    'thalassemia': thalassemia,
                })

                # Print the converted values for debugging
                print(f"Converted heart disease fields: {record_data}")

            except Exception as e:
                print(f"Error converting heart disease fields: {e}")
                # If conversion fails, try with original values
                record_data.update({
                    'chest_pain_type': cp,
                    'resting_bp': trestbps,
                    'cholesterol': chol,
                    'fasting_blood_sugar': fbs == 'Yes',
                    'rest_ecg': restecg,
                    'max_heart_rate': thalach,
                    'exercise_induced_angina': exang == 'Yes',
                    'st_depression': oldpeak,
                    'st_slope': slope,
                    'num_major_vessels': ca,
                    'thalassemia': thal,
                })
        elif prediction_type == 'kidney':
            # Add kidney-specific fields when implemented
            pass

        # Print the record data for debugging
        print(f"Record data to be saved: {record_data}")

        # Create and save the record
        serializer = self.get_serializer(data=record_data)

        # Check if the serializer is valid
        if not serializer.is_valid():
            print(f"Serializer validation errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Save the record
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    # Add custom actions for each prediction type
    @action(detail=False, methods=['post'], url_path='save')
    def save_prediction(self, request):
        print(f"Received prediction data: {request.data}")
        try:
            response = self.create(request)
            print(f"Created prediction with response: {response.data}")
            return response
        except Exception as e:
            print(f"Error creating prediction: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
