from flask import Flask, jsonify, render_template, Response
import time
import json
from datetime import datetime
from queries import (
    get_table_info,
    get_booking_info,
)

app = Flask(__name__)

@app.route("/")
def index():
    # Provide initial dummy data for rendering the template
    state_data = {
        'reward': 0,
        'action': 'None',
        'stage': 0,
        'episode': 0
    }

    weather_data = {
        'weather_type': 'Unknown',
        'car_coef': 0.0,
        'train_coef': 0.0,
        'plane_coef': 0.0,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    booking_data = {
        'transport': 'Unknown',
        'status': 'Unknown',
    }

    return render_template("index.html", current_state=state_data, current_weather=weather_data, current_booking=booking_data)

@app.route("/current_state", methods=["GET"])
def get_current_state():
    state = get_table_info("public.agent_info")
    weather = get_table_info("public.weather_info")
    transport, status = get_booking_info()

    state_data = {
        'reward': state[1],
        'action': state[2],
        'stage': state[3],
        'episode': state[4]
    }

    weather_data = {
        'weather_type': weather[0],
        'car_coef': weather[1],
        'train_coef': weather[2],
        'plane_coef': weather[3],
        'time': weather[4].strftime('%Y-%m-%d %H:%M:%S')
    }

    booking_data = {
        'transport': transport if transport else 'None',
        'status': status if status else 'None',
    }

    return jsonify(current_state=state_data, current_weather=weather_data, current_booking=booking_data)


@app.route("/stream")
def stream():
    def event_stream():
        while True:
            state = get_table_info("public.agent_info")
            weather = get_table_info("public.weather_info")
            transport, status = get_booking_info("public.booking_info")

            state_data = {
                'reward': state[1],
                'action': state[2],
                'stage': state[3],
                'episode': state[4]
            }

            weather_data = {
                'weather_type': weather[0],
                'car_coef': weather[1],
                'train_coef': weather[2],
                'plane_coef': weather[3],
                'time': weather[4].strftime('%Y-%m-%d %H:%M:%S')
            }

            booking_data = {
                'transport': transport if transport else 'None',
                'status': status if status else 'None',
            }

            yield f"data: {json.dumps({'current_state': state_data, 'current_weather': weather_data, 'current_booking': booking_data})}\n\n"
            time.sleep(0.1)

    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)
