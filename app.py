from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Main project folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Saved machine-learning model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "tuned_random_forest_hotel_cancellation_model.pkl"
)

# Find the HTML file inside the templates folder automatically
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

HTML_PATH = next(
    os.path.join(TEMPLATES_DIR, file_name)
    for file_name in os.listdir(TEMPLATES_DIR)
    if os.path.isfile(os.path.join(TEMPLATES_DIR, file_name))
)

# Load the trained model
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        hotel = request.form["hotel"]
        lead_time = int(request.form["lead_time"])
        arrival_date_year = int(request.form["arrival_date_year"])
        arrival_date_month = request.form["arrival_date_month"]
        arrival_date_week_number = int(request.form["arrival_date_week_number"])
        arrival_date_day_of_month = int(request.form["arrival_date_day_of_month"])
        stays_in_weekend_nights = int(request.form["stays_in_weekend_nights"])
        stays_in_week_nights = int(request.form["stays_in_week_nights"])
        adults = int(request.form["adults"])
        children = int(request.form["children"])
        babies = int(request.form["babies"])
        meal = request.form["meal"]
        country = request.form["country"].upper()
        market_segment = request.form["market_segment"]
        distribution_channel = request.form["distribution_channel"]
        is_repeated_guest = int(request.form["is_repeated_guest"])
        previous_cancellations = int(request.form["previous_cancellations"])

        previous_bookings_not_canceled = int(
            request.form["previous_bookings_not_canceled"]
        )

        reserved_room_type = request.form["reserved_room_type"]
        assigned_room_type = request.form["assigned_room_type"]
        deposit_type = request.form["deposit_type"]
        days_in_waiting_list = int(request.form["days_in_waiting_list"])
        customer_type = request.form["customer_type"]
        adr = float(request.form["adr"])

        required_car_parking_spaces = int(
            request.form["required_car_parking_spaces"]
        )

        total_of_special_requests = int(
            request.form["total_of_special_requests"]
        )

        # Create the same features used in the Jupyter Notebook
        total_nights = stays_in_weekend_nights + stays_in_week_nights
        total_guests = adults + children + babies
        is_family = int(children > 0 or babies > 0)
        total_revenue = adr * total_nights

        # Create one booking row for prediction
        booking_data = pd.DataFrame([{
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": arrival_date_year,
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": arrival_date_week_number,
            "arrival_date_day_of_month": arrival_date_day_of_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type,
            "assigned_room_type": assigned_room_type,
            "deposit_type": deposit_type,
            "days_in_waiting_list": days_in_waiting_list,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
            "total_nights": total_nights,
            "total_guests": total_guests,
            "is_family": is_family,
            "total_revenue": total_revenue
        }])

        # Prediction
        prediction = model.predict(booking_data)[0]

        probability = round(
            model.predict_proba(booking_data)[0][1] * 100,
            2
        )

        if prediction == 1:
            result = "This booking is likely to be cancelled."
        else:
            result = "This booking is unlikely to be cancelled."

    # Read separate frontend file
    with open(HTML_PATH, "r", encoding="utf-8") as file:
        html_code = file.read()

    return render_template_string(
        html_code,
        result=result,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)