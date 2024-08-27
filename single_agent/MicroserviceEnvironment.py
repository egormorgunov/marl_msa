from queries import (
    get_info,
    insert_booking_info,
    clear_table_info,
)

class RoadEnvironment:
    def __init__(self):
        self.weather_coefficient1 = -0.5
        self.weather_coefficient2 = -0.2
        self.weather_coefficient3 = -0.4
        self.total_reward = 0
        self.current_stage = 0
        self.stages = [
            {"name": "Start", "actions": [0, 1], "description": ["Take order"]},
            {"name": "Order taken", "actions": [0, 1], "description": ["Collect weather data", "Return"]},
            {"name": "Weather data collected", "actions": [0, 1, 2], "description": ["Choose route 1", "Choose route 2", "Choose route 3"]},
            {"name": "Route chosen", "actions": [0, 1], "description": ["Complete order", "Return"]},
            {"name": "Order completed", "actions": [0, 1], "description": ["Restart"]},
        ]
        self.weather_id, self.car_coef, self.train_coef, self.plane_coef = 1, 1, 1, 1
        self.action1_counter = 0
        self.action2_counter = 0
        self.action3_counter = 0
        self.action4_counter = 0

    def reset(self):
        self.current_stage = 0
        self.weather_id = 0
        action = 0
        state = [self.current_stage, action, self.weather_id]
        self.total_reward = 0
        print("Environment reset.")
        self.print_current_stage()
        return state

    def step(self, action):
        reward = -1
        if not self.is_done():
            if self.current_stage == 0:
                if action == 0 or action == 1:
                    self.current_stage = 1
                    print("made 1 step")
                elif action == 2:
                    print('Service waits')
            elif self.current_stage == 1:
                if action == 0:
                    self.current_stage = 2
                    self.weather_id, self.car_coef, self.train_coef, self.plane_coef = get_info()
                    if self.weather_id == "Sun":
                        self.weather_id = 0
                    elif self.weather_id == "Rain":
                        self.weather_id = 1
                    print("COEFFICIENTS = ", self.weather_id, self.car_coef, self.train_coef, self.plane_coef)
                    print("made 2 step to collect weather")
                elif action == 1:
                    self.current_stage = 0
                    print("made 2 step to return")
                elif action == 2:
                    print('Service waits')
            elif self.current_stage == 2:
                if action == 0:
                    self.current_stage = 3
                    reward = self.weather_coefficient1 * self.car_coef
                    self.action1_counter += 1
                    print(f"made 3 step, road {action + 1} selected")
                elif action == 1:
                    self.current_stage = 4
                    reward = self.weather_coefficient2 * self.train_coef
                    self.action2_counter += 1
                    print(f"made 3 step, road {action + 1} selected")
                elif action == 2:
                    self.current_stage = 5
                    reward = self.weather_coefficient3 * self.plane_coef
                    self.action3_counter += 1
                    print(f"made 3 step, road {action + 1} selected")
            elif self.current_stage == 3:
                self.transport = 'Car'
                if action == 0:
                    self.current_stage = 6
                    reward = -3
                    print('skipped booking')
                elif action == 1:
                    self.current_stage = 6
                    self.status = 'Active'
                    self.action4_counter += 1
                    reward = -1
                    insert_booking_info(self.transport, self.status, "public.booking_info")
                    print('booking successful')
                elif action == 2:
                    self.current_stage = 6
                    self.status = 'Inactive'
                    reward = -2
                    insert_booking_info(self.transport, self.status, "public.booking_info")
                    print('booking incorrect')
            elif self.current_stage == 4:
                self.transport = 'Train'
                if action == 0:
                    self.current_stage = 6
                    reward = -3
                    print('skipped booking')
                elif action == 1:
                    self.current_stage = 6
                    self.status = 'Active'
                    reward = -1
                    self.action4_counter += 1
                    insert_booking_info(self.transport, self.status, "public.booking_info")
                    print('booking successful')
                elif action == 2:
                    self.current_stage = 6
                    self.status = 'Inactive'
                    reward = -2
                    insert_booking_info(self.transport, self.status, "public.booking_info")
                    print('booking incorrect')
            elif self.current_stage == 5:
                self.transport = 'Plane'
                if action == 0:
                    self.current_stage = 6
                    reward = -3
                    print('skipped booking')
                elif action == 1:
                    self.current_stage = 6
                    self.status = 'Active'
                    self.action4_counter += 1
                    reward = -1
                    insert_booking_info(self.transport, self.status, "public.booking_info")
                    print('booking successful')
                elif action == 2:
                    self.current_stage = 6
                    self.status = 'Inactive'
                    reward = -2
                    insert_booking_info(self.transport, self.status, "public.booking_info")
                    print('booking incorrect')
            elif self.current_stage == 6:
                if action == 0:
                    self.current_stage = 7
                    print("made 4 step")
                elif action == 1:
                    self.current_stage = 5
                    clear_table_info("public.booking_info")
                    print("made 4 step to return")
                elif action == 2:
                    print('Service waits')
            elif self.current_stage == 7:
                if action == 0 or action == 1 or action == 2:
                    print("made 5 step")
                    self.is_done()
                    return True

        self.total_reward += reward
        self.total_reward = round(self.total_reward, 2)
        state = [self.current_stage, action, self.weather_id]
        print("State = ", state)
        return reward, state, self.is_done(), self.action1_counter, self.action2_counter, self.action3_counter, self.action4_counter

    def is_done(self):
        return self.current_stage == 7

    def print_current_stage(self):
        stage_info = self.stages[self.current_stage]
        print(f"Current stage: {stage_info['name']}")
        print(f"Available actions: {', '.join(stage_info['description'])}")
        print(f"Total reward: {self.total_reward} \n")



# import random

# env = RoadEnvironment()
#
# num_episodes = 5
# possible_actions = [0, 1]
#
# for episode in range(num_episodes):
#     print(f"Episode {episode + 1}")
#     env.reset()
#     done = False
#     while not env.is_done():
#         action = random.choice(possible_actions)
#         reward, state, done = env.step(action)
#         print('INFO: ', reward, state, done, ' \n')
