import numpy as np
from tinydb import TinyDB, Query
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
db = TinyDB('robot.json')
Robot = Query()
B = 0.34   # tread = diameter (34 cm)

class IntSensor(BaseModel): # interoceptive sensor
    left:     float   # left wheel delta
    right:    float   # right wheel delta


class NewRobot(BaseModel): # for create_robot
    pose: list[float]
    sigma: list[float]
    
    

def read_db_robot(id: int):
    r = db.search(Robot.id == id)
    if len(r) == 0:
        return None
    return r[0]


@app.get("/robots/{id}")
def read_robot(id: int):
    return read_db_robot(id)


@app.put("/robots/{id}")
def create_robot(id: int, init_val: NewRobot):
    sigma = [
        [init_val.sigma[0], 0, 0],
        [0, init_val.sigma[1], 0],
        [0, 0, init_val.sigma[2]],
    ]
    robot = {
        'id': id,
        'pose': init_val.pose,
        'sigma': sigma
    }
    db.remove(Robot.id == id)
    db.insert(robot)
    print(f'create: {robot}')
    return "ok"


@app.post("/robots/{id}/pred_update")
def update_robot(id: int, sensor: IntSensor):
    dsl = sensor.left
    dsr = sensor.right
    robot = read_db_robot(id)
    pose = robot['pose']
    th = pose[2][0]
    dth = (dsr - dsl) / B
    r = (dsr + dsl) / (2 * dth)
    dx = (dsr + dsl) / 2 * np.cos(th + (dsr - dsl) / (2 * B))
    dy = (dsr + dsl) / 2 * np.sin(th + (dsr - dsl) / (2 * B))
    pose[0][0] += dx
    pose[1][0] += dy
    pose[2][0] += dth
    robot['pose'] = pose
    db.remove(Robot.id == id)
    db.insert(robot)
    return "ok"
