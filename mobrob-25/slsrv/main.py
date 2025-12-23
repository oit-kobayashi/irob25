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
    s_rr:     float   # sigma_rr (right variance)
    s_ll:     float   # sigma_ll (left variance)

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
    # update pose
    pose = robot['pose']
    th = pose[2][0]
    phi = (dsr - dsl) / (2 * B)  # sin, cos の中が一緒なので
    dth = (dsr - dsl) / B
    r = (dsr + dsl) / (2 * dth)
    dx = (dsr + dsl) / 2 * np.cos(phi)
    dy = (dsr + dsl) / 2 * np.sin(phi)
    pose[0][0] += dx
    pose[1][0] += dy
    pose[2][0] += dth
    # update sigma
    sigma_p = np.array(robot['sigma'], dtype=np.float32)
    sigma_s = np.array([
        [sensor.s_rr, 0],
        [0, sensor.s_ll]
    ], dtype=np.float32)
    jp = np.array([
        [1, 0, -(dsr + dsl) / 2 * np.sin(phi)],
        [0, 1, (dsr + dsl) / 2 * np.cos(phi)],
        [0, 1, 1],   
    ], dtype=np.float32)
    js = np.array([
        [
            np.cos(phi) / 2 - (dsr + dsl) / (4 * B) * np.sin(phi),
            np.cos(phi) / 2 + (dsr + dsl) / (4 * B) * np.sin(phi)
        ],
        [
            np.sin(phi) / 2 + (dsr + dsl) / (4 * B) * np.cos(phi),
            np.sin(phi) / 2 - (dsr + dsl) / (4 * B) * np.cos(phi)
        ],
        [1 / B, -1 / B],
    ], dtype=np.float32)
    sigma_p = jp.dot(sigma_p).dot(jp.T) + js.dot(sigma_s).dot(js.T)
    # store to DB
    robot['pose'] = pose
    robot['sigma'] = sigma_p
    db.remove(Robot.id == id)
    db.insert(robot)
    return "ok"
