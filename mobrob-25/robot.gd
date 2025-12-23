extends Area2D
@export var vl = 0.0000001
@export var vr = 0
var dsr = 0.00001  # Delta S_r
var dsl = 0  # Delta S_l
var srr = 0  # sigma_rr
var sll = 0  # sigma_ll


func _ready() -> void:
	var rob = {
		'pose': [1, 2, 3],
		'sigma': [0.1 ** 2, 0.1 ** 2, 0.1 ** 1]
	}
	$HTTPRequestCreate.request(
		'http://127.0.0.1:8000/robots/123',
		['Content-Type: application/json'],
		HTTPClient.METHOD_PUT,
		JSON.stringify(rob)
	)
	pass # Replace with function body.


func _process(delta: float) -> void:
	var th = -rotation
	var sr = vr * delta
	var sl = vl * delta
	var dth = (sr - sl) / 0.34
	var r = (sr + sl) / 2 / dth
	var dx = 2 * r * sin(dth / 2) * cos(th + dth / 2)
	var dy = 2 * r * sin(dth / 2) * sin(th + dth / 2)
	dsr += sr
	dsl += sl
	srr += abs(sr) * 1.0
	sll += abs(sl) * 1.0
	
	rotation -= dth
	position.x += dx * 100
	position.y -= dy * 100
	
	if Input.is_action_pressed("FORWARD"):
		vl += 1 * delta
		vr += 1 * delta
	if Input.is_action_pressed("BACKWARD"):
		vl -= 1 * delta
		vr -= 1 * delta
	if Input.is_action_pressed("ROT_CW"):
		vl += 1 * delta
		vr -= 1 * delta
	if Input.is_action_pressed("ROT_CCW"):
		vl -= 1 * delta
		vr += 1 * delta

func _on_robot_timer_timeout() -> void:
	$HTTPRequestCreate.request(
		'http://127.0.0.1:8000/robots/123/pred_update',
		['Content-Type: application/json'],
		HTTPClient.METHOD_POST,
		JSON.stringify({
			'left': dsl,
			'right': dsr,
			's_ll': sll,
			's_rr': srr
		})
	)
	dsl = 0
	dsr = 0
	sll = 0
	srr = 0
	
