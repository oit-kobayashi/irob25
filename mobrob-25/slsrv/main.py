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

class ExtSensor(BaseModel): # exteroceptive sensor
    # note: all those params are ext. sensor's
    x:        float   # x position
    y:        float   # y position
    th:       float   # theta
    s_xx:     float   # sigma_xx
    s_yy:     float   # sigma_yy
    s_tt:     float   # sigma_theta_theta
    
class NewRobot(BaseModel): # for create_robot
    pose: list[float]
    sigma: list[float]
    
    

def read_db_robot(id: int):
    r = db.search(Robot.id == id)
    if len(r) == 0:
        return None
    return r[0]


# タプルを返す (固有値リスト, 固有ベクトルリスト)
# ex. ([1, 2], [[1, 2], [3, 4]]) λ=1に対して[1,2]
def calc_eig(sigma: np.ndarray) -> tuple[list[float], list[list[float]]]:
    lmd, vec = np.linalg.eig(sigma[0:2, 0:2])
    return (lmd.tolist(), vec.T.tolist())


@app.get("/robots/{id}")
def read_robot(id: int):
    print("read_robot")
    return read_db_robot(id)


@app.put("/robots/{id}")
def create_robot(id: int, init_val: NewRobot):
    sigma = [
        [init_val.sigma[0], 0, 0],
        [0, init_val.sigma[1], 0],
        [0, 0, init_val.sigma[2]],
    ]
    eigvals, eigvecs = calc_eig(np.array(sigma))
    robot = {
        'id': id,
        'pose': init_val.pose,
        'sigma': sigma,
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs
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
    th = pose[2]
    phi = th + (dsr - dsl) / (2 * B)  # sin, cos の中が一緒なので
    dth = (dsr - dsl) / B
    r = (dsr + dsl) / (2 * dth)
    dx = (dsr + dsl) / 2 * np.cos(phi) # 1st order approx.
    dy = (dsr + dsl) / 2 * np.sin(phi) # 1st order approx.
    # dx = 2 * r * np.sin(dth / 2) * np.cos(th + dth / 2) # accurate
    # dy = 2 * r * np.sin(dth / 2) * np.sin(th + dth / 2) # accurate
    pose[0] += dx
    pose[1] += dy
    pose[2] += dth
    # update sigma
    sigma_p = np.array(robot['sigma'], dtype=np.float32)
    sigma_s = np.array([
        [sensor.s_rr, 0],
        [0, sensor.s_ll]
    ], dtype=np.float32)
    jp = np.array([
        [1, 0, -(dsr + dsl) / 2 * np.sin(phi)],
        [0, 1, (dsr + dsl) / 2 * np.cos(phi)],
        [0, 0, 1],   
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
    eigvals, eigvecs = calc_eig(sigma_p)
    robot['pose'] = pose
    robot['sigma'] = sigma_p.tolist()
    robot['eigenvalues'] = eigvals
    robot['eigenvectors'] = eigvecs
    # print(robot)
    db.remove(Robot.id == id)
    db.insert(robot)
    return "ok"


@app.post("/robots/{id}/perc_update")
def update_robot(id: int, sensor: ExtSensor):
    print("perception update")
    robot = read_db_robot(id)
    # robot's opinion
    p1 = np.array(robot['pose'], dtype=np.float32).reshape((3,1))
    s1 = np.array(robot['sigma'], dtype=np.float32)
    # ext. sensor's opinion
    p2 = np.array([[sensor.x],
                   [sensor.y],
                   [sensor.th]])
    s2 = np.array([[sensor.s_xx, 0, 0],
                   [0, sensor.s_yy, 0],
                   [0, 0, sensor.s_tt]])
    # Kalman filter
    k = s1.dot(np.linalg.inv(s1 + s2))
    print(k)
    print(p2)
    print(p1)
    print(p2-p1)
    p1 += k.dot(p2 - p1)
    print(f'before: {s1}')
    s1 += -k.dot(s1 + s2).dot(k.T)
    print(f'after: {s1}')

    # store to DB
    robot['pose'] = p1.reshape((3,)).tolist()
    robot['sigma'] = s1.tolist()
    db.remove(Robot.id == id)
    db.insert(robot)
    return "ok"
