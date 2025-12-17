extends Area2D
@export var vl = 0.0000001
@export var vr = 0


func _ready() -> void:
	var rob = {
		'pose': [1, 2, 3],
		'sigma': [4, 5, 6]
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
	var dsr = vr * delta
	var dsl = vl * delta
	var dth = (dsr - dsl) / 0.34
	var r = (dsr + dsl) / 2 / dth
	var dx = 2 * r * sin(dth / 2) * cos(th + dth / 2)
	var dy = 2 * r * sin(dth / 2) * sin(th + dth / 2)
	
	rotation -= dth
	position.x += dx * 100
	position.y -= dy * 100
	
	if Input.is_action_pressed("FORWARD"):
		vl += 1 * delta
		vr += 1 * delta
	if Input.is_action_pressed("ROT_CW"):
		vl += 1 * delta
		vr -= 1 * delta
	if Input.is_action_pressed("ROT_CCW"):
		vl -= 1 * delta
		vr += 1 * delta
