extends Area2D
@export var vl = 0.0000001
@export var vr = 0
var dsr = 0.00001  # Delta S_r
var dsl = 0  # Delta S_l
var srr = 0  # sigma_rr
var sll = 0  # sigma_ll
var eig_vals = [0, 0]
var eig_vecs = [[1, 0], [0, 1]]
var est_pose = [3.0, -1.0, -3.141592/2]  # also initial pose
var ext_sensor_count = 10 # exteroceptive sensor counter


func _ready() -> void:
	position = Vector2(est_pose[0] * 100, -est_pose[1] * 100)
	rotation = -est_pose[2]
	var rob = {
		'pose': est_pose,
		'sigma': [0.1 ** 2, 0.1 ** 2, 0.1 ** 1]
	}
	$HTTPRequest.request(
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
	srr += abs(sr) * 0.001
	sll += abs(sl) * 0.001
	
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
	queue_redraw()

func _on_robot_timer_timeout() -> void:
	$HTTPRequest.request(
		'http://127.0.0.1:8000/robots/123/pred_update',
		['Content-Type: application/json'],
		HTTPClient.METHOD_POST,
		JSON.stringify({
			'left': dsl + randfn(0, sll),
			'right': dsr + randfn(0, srr),
			's_ll': sll,
			's_rr': srr
		})
	)
	dsl = 0
	dsr = 0
	sll = 0
	srr = 0
	
func _draw() -> void:
	var r1 = eig_vals[0] ** 0.5
	var r2 = eig_vals[1] ** 0.5
	var ph = atan2(eig_vecs[0][1], eig_vecs[0][0])
	var _est_pose = Vector2(   # est_pose on screen coordinate
		est_pose[0] * 100,
		-est_pose[1] * 100
	) - position
	var mat = Transform2D()
	mat = mat.scaled(Vector2(r1, r2) * 10).rotated(-ph).translated(_est_pose).rotated(-rotation)
	draw_set_transform_matrix(mat)
	draw_circle(Vector2(0, 0), 30, Color.AQUA, false)
	draw_circle(Vector2(0, 0), 1, Color.WHITE, false)
	draw_set_transform_matrix(Transform2D.IDENTITY)
	

func _on_http_request_get_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	var j = JSON.parse_string(body.get_string_from_utf8())
	est_pose = j['pose']
	eig_vals = j['eigenvalues']
	eig_vecs = j['eigenvectors']
	ext_sensor_count -= 1
	if ext_sensor_count <= 0:
		ext_sensor_count = 10
		print("perception update!")
		$HTTPRequest.request(
			'http://127.0.0.1:8000/robots/123/perc_update',
			['Content-Type: application/json'],
			HTTPClient.METHOD_POST,
			JSON.stringify({
				'x': randfn(position.x / 100, 1),
				'y': randfn(-position.y / 100, 1),
				'th': randfn(-rotation, 0.1),
				's_xx': 1.0,
				's_yy': 1.0,
				's_tt': 0.1
			})
		)



func _on_http_request_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	$HTTPRequestGet.request('http://127.0.0.1:8000/robots/123')


func _on_http_request_perception_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	var j = JSON.parse_string(body.get_string_from_utf8())
	est_pose = j['pose']
	
