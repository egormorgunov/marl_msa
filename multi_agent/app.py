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
        'reward1': 0,
        'action1': 'None',
        'stage1': 0,
        'episode': 0,
        'reward2': 0,
        'action2': 'None',
        'stage2': 0,
    }

    weather_data = {
        'weather_type': 'Unknown',
        'car_coef': 0.0,
        'train_coef': 0.0,
        'plane_coef': 0.0,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    booking_data = {
        'transport1': 'Unknown',
        'status1': 'Unknown',
        'transport2': 'Unknown',
        'status2': 'Unknown',
    }

    return render_template("index.html", current_state=state_data, current_weather=weather_data, current_booking=booking_data)


@app.route("/current_state", methods=["GET"])
def get_current_state():
    state1 = get_table_info("public.agent1_info")
    state2 = get_table_info("public.agent2_info")
    weather = get_table_info("public.weather_info")
    transport1, status1 = get_booking_info("public.booking1_info")
    transport2, status2 = get_booking_info("public.booking2_info")

    state_data = {
        'reward1': state1[1],
        'action1': state1[2],
        'stage1': state1[3],
        'episode': state1[4],
        'reward2': state2[1],
        'action2': state2[2],
        'stage2': state2[3],
    }

    weather_data = {
        'weather_type': weather[0],
        'car_coef': weather[1],
        'train_coef': weather[2],
        'plane_coef': weather[3],
        'time': weather[4].strftime('%Y-%m-%d %H:%M:%S')
    }

    booking_data = {
        'transport1': transport1 if transport1 else 'None',
        'status1': status1 if status1 else 'None',
        'transport2': transport2 if transport2 else 'None',
        'status2': status2 if status2 else 'None',
    }

    return jsonify(current_state=state_data, current_weather=weather_data, current_booking=booking_data)


@app.route("/stream")
def stream():
    def event_stream():
        while True:
            state1 = get_table_info("public.agent1_info")
            state2 = get_table_info("public.agent2_info")
            weather = get_table_info("public.weather_info")
            transport1, status1 = get_booking_info("public.booking1_info")
            transport2, status2 = get_booking_info("public.booking2_info")

            state_data = {
                'reward1': state1[1],
                'action1': state1[2],
                'stage1': state1[3],
                'episode': state1[4],
                'reward2': state2[1],
                'action2': state2[2],
                'stage2': state2[3],
            }

            weather_data = {
                'weather_type': weather[0],
                'car_coef': weather[1],
                'train_coef': weather[2],
                'plane_coef': weather[3],
                'time': weather[4].strftime('%Y-%m-%d %H:%M:%S')
            }

            booking_data = {
                'transport1': transport1 if transport1 else 'None',
                'status1': status1 if status1 else 'None',
                'transport2': transport2 if transport2 else 'None',
                'status2': status2 if status2 else 'None',
            }

            yield f"data: {json.dumps({'current_state': state_data, 'current_weather': weather_data, 'current_booking': booking_data})}\n\n"
            time.sleep(0.1)

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
